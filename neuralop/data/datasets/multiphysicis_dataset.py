import torch
import torch.nn.functional as F
import numpy as np
import h5py

from pathlib import Path
from typing import List, Union
from scipy.interpolate import RegularGridInterpolator

from .tensor_dataset import TensorDataset
from ..transforms.data_processors import MultiphysicsDataProcessor
from ..transforms.normalizers import MultiphysicsUnitGaussianNormalizer


def load_data(root_dir, dataset_name, process_type, resolution, file_format=".pt"):
    file_path = Path(root_dir).joinpath(f"{dataset_name}_{process_type}_16{file_format}")

    if file_format == ".pt":
        data = torch.load(file_path.as_posix())
        return data
    elif file_format == ".h5":
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            print(f"File {file_path} includes: {keys}")
            data = {key: f[key][:] for key in keys}
        return data
    else:
        raise ValueError(f"Unknown file format: {file_format}")


# class GridPreprocessor:
#     """ GridPreprocessor resizes and subsample raw data
#     """
#
#     def resize_to_common_grid_batch(self, data, target_grid_shape):
#         _, *dim_shapes = data.shape
#         x, y = [np.linspace(0, 1, dim_i) for dim_i in dim_shapes]
#         interpolator_new = RegularGridInterpolator(
#             (x, y), data
#         )
#         grid_new = np.stack(
#             list(np.meshgrid(*(np.linspace(0, 1, target_grid_shape) for _ in range(len(dim_shapes))), indexing='ij')),
#             axis=-1
#         )
#         return interpolator_new(grid_new)
#
#     def get_mode(self, data):
#         # `nearest`,
#         # `linear` (3D-only),
#         # `bilinear`,
#         # `bicubic` (4D-only),
#         # `trilinear` (5D-only),
#
#         if data.dtype is torch.bool:
#             return 'nearest'
#
#         if data.shape != 5:
#             return 'bilinear'
#         else:
#             return 'trilinear'
#
#     def resize_to_common_grid(self, data, target_resolution, interpolate_mode):
#         original_shape = data.shape
#         original_ndim = data.ndim
#
#         if torch.all(torch.tensor(original_shape[1:]) == torch.tensor(target_resolution)):
#             return data
#
#         # 4D (B, C, H, W)
#         if original_ndim == 2:
#             # (B, X) -> (B, 1, 1, X)
#             data = data.unsqueeze(1).unsqueeze(2)
#         elif original_ndim == 3:
#             # (B, X, Y) -> (B, 1, X, Y)
#             data = data.unsqueeze(1)
#
#         if interpolate_mode is None:
#             interpolate_mode = self.get_mode(data)
#
#         data_resized = F.interpolate(
#             data.float(),
#             size=target_resolution,
#             mode=interpolate_mode,
#         )
#
#         if original_ndim == 2:
#             data_resized = data_resized.squeeze(1).squeeze(2)
#         elif original_ndim == 3:
#             data_resized = data_resized.squeeze(1)
#
#         return data_resized
#
#     def subsample(self, data, subsampling_rate, n_train, channel_dim):
#         data_dims = data.ndim - 2
#
#         if not subsampling_rate:
#             subsampling_rate = 1
#         if not isinstance(subsampling_rate, list):
#             subsampling_rate = [subsampling_rate] * data_dims
#         assert len(subsampling_rate) == data_dims, \
#             f"Error: length mismatch between input_subsampling_rate and dimensions of data.\
#                         input_subsampling_rate must be one int shared across all dims, or an iterable of\
#                             length {len(data_dims)}, got {subsampling_rate}"
#
#         train_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in subsampling_rate]
#         train_indices.insert(channel_dim, slice(None))
#         return data[train_indices]


def resize_to_common_grid_batch(data, target_grid_shape):
    _, *dim_shapes = data.shape
    x, y = [np.linspace(0, 1, dim_i) for dim_i in dim_shapes]
    interpolator_new = RegularGridInterpolator(
        (x, y), data
    )
    grid_new = np.stack(
        list(np.meshgrid(*(np.linspace(0, 1, target_grid_shape) for _ in range(len(dim_shapes))), indexing='ij')),
        axis=-1
    )
    return interpolator_new(grid_new)


def get_mode(data):
    # `nearest`,
    # `linear` (3D-only),
    # `bilinear`,
    # `bicubic` (4D-only),
    # `trilinear` (5D-only),

    if data.dtype is torch.bool:
        return 'nearest'

    if data.shape != 5:
        return 'bilinear'
    else:
        return 'trilinear'


def resize_to_common_grid(data, target_resolution, interpolate_mode):
    original_shape = data.shape
    original_ndim = data.ndim

    if torch.all(torch.tensor(original_shape[1:]) == torch.tensor(target_resolution)):
        return data

    # 4D (B, C, H, W)
    if original_ndim == 2:
        # (B, X) -> (B, 1, 1, X)
        data = data.unsqueeze(1).unsqueeze(2)
    elif original_ndim == 3:
        # (B, X, Y) -> (B, 1, X, Y)
        data = data.unsqueeze(1)

    if interpolate_mode is None:
        interpolate_mode = get_mode(data)

    data_resized = F.interpolate(
        data.float(),
        size=target_resolution,
        mode=interpolate_mode,
    )

    if original_ndim == 2:
        data_resized = data_resized.squeeze(1).squeeze(2)
    elif original_ndim == 3:
        data_resized = data_resized.squeeze(1)

    return data_resized


def subsample(data, subsampling_rate, n_train, channel_dim):
    data_dims = data.ndim - 2

    if not subsampling_rate:
        subsampling_rate = 1
    if not isinstance(subsampling_rate, list):
        subsampling_rate = [subsampling_rate] * data_dims
    assert len(subsampling_rate) == data_dims, \
        f"Error: length mismatch between input_subsampling_rate and dimensions of data.\
                    input_subsampling_rate must be one int shared across all dims, or an iterable of\
                        length {len(data_dims)}, got {subsampling_rate}"

    train_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in subsampling_rate]
    train_indices.insert(channel_dim, slice(None))
    return data[train_indices]


class MultiphysicsDataset:
    """MultiphysicsDataset is a base Dataset class for our library.
            MultiphysicsDataset contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            datasets may implement a download flag at init, which provides
            access to a number of premade datasets for sample problems provided
            in our Zenodo archive.

        All datasets are required to expose the following attributes after init:

        train_db: torch.utils.data.Dataset of training examples
        test_db:  ""                       of test examples
        data_processor: neuralop.data.transforms.DataProcessor to process data examples
            optional, default is None
        """

    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name: str,
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: List[int],
                 encode_input: bool = False,
                 encode_output: bool = True,
                 encoding="channel-wise",
                 interpolate_mode: str = None,
                 input_subsampling_rate=None,
                 output_subsampling_rate=None,
                 channel_dim=1,
                 channels_squeezed=True):
        """MultiphysicsDataset

        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_train : int
            number of train instances
        n_tests : List[int]
            number of test instances per test dataset
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        channels_squeezed : bool, optional
            If the channels dim is 1, whether that is explicitly kept in the saved tensor.
            If not, we need to unsqueeze it to explicitly have a channel dim.
            Only applies when there is only one data channel, as in our example problems
            Defaults to True
        """

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes

        # Load train data
        data = load_data(root_dir, dataset_name, 'train', train_resolution, file_format='.pt')

        data["x"] = resize_to_common_grid(data["x"], train_resolution, interpolate_mode)
        data["y"] = resize_to_common_grid(data["y"], train_resolution, interpolate_mode)

        x_train = data["x"].type(torch.float32).clone()
        y_train = data["y"].type(torch.float32).clone()

        if channels_squeezed:
            x_train = x_train.unsqueeze(channel_dim)
            y_train = y_train.unsqueeze(channel_dim)

        x_train = subsample(x_train, input_subsampling_rate, n_train, channel_dim)
        y_train = subsample(y_train, output_subsampling_rate, n_train, channel_dim)

        del data

        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            if not hasattr(self, 'input_normalizer'):
                self.input_encoder = MultiphysicsUnitGaussianNormalizer()

            self.input_encoder.add_task(dataset_name, dim=reduce_dims)
            self.input_encoder.set_task(dataset_name)
            self.input_encoder.fit(x_train)
        else:
            self.input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            if not hasattr(self, 'output_normalizer'):
                self.output_encoder = MultiphysicsUnitGaussianNormalizer()

            self.output_encoder.add_task(dataset_name, dim=reduce_dims)
            self.output_encoder.set_task(dataset_name)
            self.output_encoder.fit(y_train)
        else:
            self.output_encoder = None

        # Save train dataset
        self._train_db = TensorDataset(
            x_train,
            y_train,
        )

        # create DataProcessor
        self._data_processor = MultiphysicsDataProcessor(in_normalizer=self.input_encoder,
                                                         out_normalizer=self.output_encoder)
        # Load test data
        self._test_dbs = {}
        for (res, n_test) in zip(test_resolutions, n_tests):
            print(f"Loading test db for resolution {res} with {n_test} samples ")

            data = load_data(root_dir, dataset_name, 'test', res, file_format='.pt')

            data["x"] = resize_to_common_grid(data["x"], res, interpolate_mode)
            data["y"] = resize_to_common_grid(data["y"], res, interpolate_mode)

            x_test = data["x"].type(torch.float32).clone()
            y_test = data["y"].type(torch.float32).clone()

            if channels_squeezed:
                x_test = x_test.unsqueeze(channel_dim)
                y_test = y_test.unsqueeze(channel_dim)

            x_test = subsample(x_test, input_subsampling_rate, n_test, channel_dim)
            y_test = subsample(y_test, output_subsampling_rate, n_test, channel_dim)

            del data

            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self._test_dbs[res] = test_db

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs
