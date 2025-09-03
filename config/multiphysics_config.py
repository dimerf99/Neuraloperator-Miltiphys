from typing import Dict, Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNO_Small2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class MultiphysicsOptConfig(OptimizationConfig):
    n_epochs: int = 300
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5


class BurgersDatasetConfig(ConfigBase):
    train_resolution: int = 32
    test_resolutions: List[int] = [32]
    test_batch_sizes: List[int] = [16]
    n_tests: List[int] = [400]
    spatial_length: int = 16
    temporal_length: int = 17
    temporal_subsample: Optional[int] = None
    encode_input: bool = False
    encode_output: bool = True
    include_endpoint: List[bool] = [True, False]
    download_params: Dict[str, Any] = {
        'source': 'pdebench',
        'dataset_id': '268190',
        'download': False
    }


class DarcyDatasetConfig(ConfigBase):
    train_resolution: int = 32
    test_resolutions: List[int] = [32]
    n_tests: List[int] = [100]
    test_batch_sizes: List[int] = [16]
    encode_input: bool = True
    encode_output: bool = True
    download_params: Dict[str, Any] = {
        'source': 'zenodo',
        'dataset_id': '12784353',
        'download': False
    }


class MultiphysicsDatasetConfig(ConfigBase):
    folder: str = 'neuralop/data/datasets/data/'
    batch_size: int = 8
    n_train: int = 1000
    datasets: Dict[str, Any] = {
        'darcy': DarcyDatasetConfig(),
        'burgers': BurgersDatasetConfig()
    }


class Patching(ConfigBase):
    levels: int = 1
    padding: int = 16
    stitching: bool = True


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Small2d()
    opt: OptimizationConfig = MultiphysicsOptConfig()
    data: MultiphysicsDatasetConfig = MultiphysicsDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
