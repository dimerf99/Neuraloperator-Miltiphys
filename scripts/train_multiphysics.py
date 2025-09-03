import torch

from torch.utils.data import DataLoader, DistributedSampler
import wandb

from neuralop.losses.data_losses import H1Loss, LpLoss
from neuralop.models.base_model import get_model
from neuralop.training.trainer import Trainer
from neuralop.data.datasets.multiphysics_wrapper import load_data
from neuralop.data.transforms.data_processors import MultiTaskMGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.mpu.comm import get_local_rank
from neuralop.utils import get_wandb_api_key, count_model_params, get_project_root

from zencfg import make_config_from_cli
import sys

sys.path.insert(0, '../')
from config.multiphysics_config import Default


def main():
    config = make_config_from_cli(Default)
    config = config.to_dict()

    device, is_logger = setup(config)

    wandb_args = None
    if config.wandb.log and is_logger:
        wandb.login(key=get_wandb_api_key())
        if config.wandb.name:
            wandb_name = config.wandb.name
        else:
            wandb_name = "_".join(
                f"{var}"
                for var in [
                    config.model.model_arch,
                    config.model.n_layers,
                    config.model.n_modes,
                    config.model.hidden_channels,
                ]
            )
        wandb_args = dict(
            config=config,
            name=wandb_name,
            group=config.wandb.group,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]
        wandb.init(**wandb_args)

    config.verbose = config.verbose and is_logger

    if config.verbose and is_logger:
        print(f"##### CONFIG #####\n")
        print(config)
        sys.stdout.flush()

    multiphysics_data = {}

    for physics_name in list(config.data.datasets.keys()):
        physics_config = config.data.datasets[physics_name]
        data_root = get_project_root() / config.data.folder

        task_config_full = {
            'n_train': config.data.n_train,
            'n_tests': physics_config.n_tests,
            'batch_size': config.data.batch_size,
            'train_resolution': physics_config.train_resolution,
            'test_resolutions': physics_config.test_resolutions,
            'test_batch_sizes': physics_config.test_batch_sizes,
            'interpolate_mode': physics_config.interpolate_mode,
            'data_root': data_root,
            'encode_input': physics_config.encode_input,
            'encode_output': physics_config.encode_output,
            'physics_name': physics_name,
            'download_params': physics_config.download_params,
            'temporal_subsample': physics_config.temporal_subsample,
            'spatial_subsample': physics_config.spatial_subsample,
        }

        train_loader, test_loaders, data_processor = load_data(**task_config_full)

        multiphysics_data[physics_name] = {
            'train_loader': train_loader,
            'test_loaders': test_loaders,
            'data_processor': data_processor
        }

    model = get_model(config)

    for physics_name, physics_data in multiphysics_data.items():
        if config.patching.levels > 0:
            physics_data['data_processor'] = MultiTaskMGPatchingDataProcessor(
                model=model,
                physics_config=physics_config,
                padding_fraction=config.patching.padding,
                stitching=config.patching.stitching,
                levels=config.patching.levels,
                use_distributed=config.distributed.use_distributed,
                device=device
            )

    if config.distributed.use_distributed:
        for task_name, task_data in multiphysics_data.items():
            train_db = task_data['train_loader'].dataset
            train_sampler = DistributedSampler(train_db, rank=get_local_rank())
            task_data['train_loader'] = DataLoader(
                dataset=train_db,
                batch_size=config.data.batch_size,
                sampler=train_sampler
            )
            new_test_loaders = {}
            for res, loader in task_data['test_loaders'].items():
                task_config = task_data['config']
                if 'test_batch_sizes' in task_config and 'test_resolutions' in task_config:
                    res_index = task_config['test_resolutions'].index(res)
                    batch_size = task_config['test_batch_sizes'][res_index]
                else:
                    batch_size = loader.batch_size

                test_db = loader.dataset
                test_sampler = DistributedSampler(test_db, rank=get_local_rank())
                new_test_loaders[res] = DataLoader(
                    dataset=test_db,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=test_sampler
                )
            task_data['test_loaders'] = new_test_loaders

    for task_data in multiphysics_data.values():
        if 'config' in task_data:
            del task_data['config']

    optimizer = AdamW(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )

    if config.opt.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.opt.gamma,
            patience=config.opt.scheduler_patience,
            mode="min",
        )
    elif config.opt.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt.scheduler_T_max
        )
    elif config.opt.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
        )
    else:
        raise ValueError(f"Got scheduler={config.opt.scheduler}")

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    if config.opt.training_loss == "l2":
        train_loss = l2loss
    elif config.opt.training_loss == "h1":
        train_loss = h1loss
    else:
        raise ValueError(
            f'Got training_loss={config.opt.training_loss} '
            f'but expected one of ["l2", "h1"]'
        )
    eval_losses = {"h1": h1loss, "l2": l2loss}

    if config.verbose and is_logger:
        print("\n### MODEL ###\n", model)
        print("\n### OPTIMIZER ###\n", optimizer)
        print("\n### SCHEDULER ###\n", scheduler)
        print("\n### LOSSES ###")
        print(f"\n * Train: {train_loss}")
        print(f"\n * Test: {eval_losses}")
        print(f"\n### Beginning Training...\n")
        sys.stdout.flush()

    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=data_processor,
        mixed_precision=config.opt.mixed_precision,
        wandb_log=config.wandb.log,
        eval_interval=config.opt.eval_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
    )

    # Log parameter count
    if is_logger:
        n_params = count_model_params(model)

        if config.verbose:
            print(f"\nn_params: {n_params}")
            sys.stdout.flush()

        if config.wandb.log:
            to_log = {"n_params": n_params}
            if config.n_params_baseline is not None:
                to_log["n_params_baseline"] = (config.n_params_baseline,)
                to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
                to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
            wandb.log(to_log, commit=False)
            wandb.watch(model)

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    if config.wandb.log and is_logger:
        wandb.finish()


if __name__ == '__main__':
    main()
