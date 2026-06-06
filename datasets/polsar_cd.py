from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional
from tools.data_convert.polarization_mode_convert import c3toc2
from tools.file_operation.path_process import build_imglist
from tools.data_convert.plosar_format_convert import matrix_to_sequence, sequence_to_complex, sequence_to_matrix, sequence_to_vector


class BaseDataset(Dataset):
    def __init__(self, data_path, mean=None, std=None, is_augment=False) -> None:
        pre_path = data_path.joinpath('pre')
        next_path = data_path.joinpath('next')
        gt_path = data_path.joinpath('gt')
        self.pre_imgs = build_imglist(pre_path)
        self.next_imgs = build_imglist(next_path)
        self.gt_imgs = build_imglist(gt_path)
        self.mean = np.zeros(9) if mean is None else mean
        self.std = np.ones(9) if std is None else std
        self.is_augment = is_augment

    def __len__(self):
        return len(self.gt_imgs)

    def data_process(self, pre: np.ndarray, next: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        pre_path = self.pre_imgs[index]
        next_path = self.next_imgs[index]
        gt_path = self.gt_imgs[index]
        name = Path(pre_path).stem
        pre = np.load(pre_path)
        next = np.load(next_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        self.gt = functional.to_tensor(gt)
        data = self.data_process(pre, next)
        return {'data': data, 'gt': self.gt, 'name': name}

    def augment(self, pre, next):
        pre = torch.tensor(pre, dtype=torch.float)
        next = torch.tensor(next, dtype=torch.float)
        if self.is_augment:
            if torch.rand(1) < 0.5:
                pre = functional.hflip(pre)
                next = functional.hflip(next)
                self.gt = functional.hflip(self.gt)
            if torch.rand(1) < 0.5:
                pre = functional.vflip(pre)
                next = functional.vflip(next)
                self.gt = functional.vflip(self.gt)
            if torch.rand(1) < 0.5:
                angle = np.random.randint(1, 3)*90
                pre = functional.rotate(pre, angle)
                next = functional.rotate(next, angle)
                self.gt = functional.rotate(self.gt, angle)
        return pre, next


class RealDataset(BaseDataset):
    def data_process(self, pre, next):
        compose = transforms.Compose([
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        pre, next = self.augment(pre, next)
        pre = compose(pre)
        next = compose(next)
        data = torch.cat((pre, next))
        return data


class ComplexDataset(BaseDataset):
    def data_process(self, pre, next):
        compose = transforms.Compose(
            [transforms.Normalize(mean=self.mean, std=self.std)
             ])
        pre, next = self.augment(pre, next)
        pre = sequence_to_complex(compose(pre))
        next = sequence_to_complex(compose(next))
        data = torch.cat((pre, next))
        return data


class APDataset(BaseDataset):
    def __init__(self, data_path, mean=None, std=None, is_augment=False, is_db=False) -> None:
        super().__init__(data_path, mean, std, is_augment)
        self.is_db = is_db

    def data_process(self, pre, next):
        compose = transforms.Compose(
            [transforms.Normalize(mean=self.mean, std=self.std)
             ])
        pre = sequence_to_vector(pre, self.is_db)
        next = sequence_to_vector(next, self.is_db)
        pre, next = self.augment(pre, next)
        pre = compose(pre)
        next = compose(next)
        data = torch.cat((pre, next))
        return data


class APDatasetC2(BaseDataset):
    def __init__(self, data_path, mean=None, std=None, is_augment=False, is_db=False) -> None:
        super().__init__(data_path, mean, std, is_augment)
        self.is_db = is_db

    def data_process(self, pre, next):
        compose = transforms.Compose(
            [transforms.Normalize(mean=self.mean, std=self.std)
             ])
        pre = sequence_to_vector(pre, self.is_db)
        next = sequence_to_vector(next, self.is_db)
        pre, next = self.augment(pre, next)
        pre = compose(pre)
        next = compose(next)
        pre = c3toc2(pre)
        next = c3toc2(next)
        data = torch.cat((pre, next))
        return data


class DualStreamBaseDataset(BaseDataset):
    """Dual-stream dataset returning pre and next as separate tensors.

    Instead of concatenating bi-temporal inputs along the channel dimension,
    this dataset returns them as a tuple ``(pre, next)`` in the ``'data'``
    field, suitable for Siamese-style networks with two independent encoders.
    """

    def __getitem__(self, index) -> dict:
        pre_path = self.pre_imgs[index]
        next_path = self.next_imgs[index]
        gt_path = self.gt_imgs[index]
        name = Path(pre_path).stem
        pre = np.load(pre_path)
        next = np.load(next_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        self.gt = functional.to_tensor(gt)
        pre_out, next_out = self.data_process(pre, next)
        return {'pre': pre_out, 'next': next_out, 'gt': self.gt, 'name': name}

    def data_process(self, pre, next):
        raise NotImplementedError


class DualStreamRealDataset(DualStreamBaseDataset):
    def data_process(self, pre, next):
        compose = transforms.Compose([
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        pre, next = self.augment(pre, next)
        return compose(pre), compose(next)


class DualStreamComplexDataset(DualStreamBaseDataset):
    def data_process(self, pre, next):
        compose = transforms.Compose([
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        pre, next = self.augment(pre, next)
        return sequence_to_complex(compose(pre)), sequence_to_complex(compose(next))


class DualStreamAPDataset(DualStreamBaseDataset):
    def __init__(self, data_path, mean=None, std=None, is_augment=False, is_db=False) -> None:
        super().__init__(data_path, mean, std, is_augment)
        self.is_db = is_db

    def data_process(self, pre, next):
        compose = transforms.Compose([
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        pre = sequence_to_vector(pre, self.is_db)
        next = sequence_to_vector(next, self.is_db)
        pre, next = self.augment(pre, next)
        return compose(pre), compose(next)


class DualStreamAPDatasetC2(DualStreamBaseDataset):
    def __init__(self, data_path, mean=None, std=None, is_augment=False, is_db=False) -> None:
        super().__init__(data_path, mean, std, is_augment)
        self.is_db = is_db

    def data_process(self, pre, next):
        compose = transforms.Compose([
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        pre = sequence_to_vector(pre, self.is_db)
        next = sequence_to_vector(next, self.is_db)
        pre, next = self.augment(pre, next)
        return c3toc2(compose(pre)), c3toc2(compose(next))


def get_dataset(data_path, mode: str, data_type: str, mean=None, std=None, dual_stream=False) -> Dataset:
    """Create a dataset for WHU-PolSAR-CD.

    Args:
        data_path: Root directory containing {train,valid,test}/{pre,next,gt}/
            or {pre,next,gt}/ for unsplit data.
        mode: One of 'train', 'valid', or 'test'.
        data_type: Input representation - 'real', 'complex', 'ap', 'ap_db',
            or 'ap_c2_db'.
        mean: Channel-wise mean for normalization (9-element array).
        std: Channel-wise std for normalization (9-element array).
        dual_stream: If True, return pre and next as separate tensors
            (for Siamese networks) instead of concatenating them.

    Returns:
        A PyTorch Dataset instance.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)
    split_path = data_path / mode
    if split_path.is_dir():
        data_path = split_path
    is_augment = mode == 'train'
    data_type = data_type.lower()

    if dual_stream:
        match data_type:
            case 'complex':
                return DualStreamComplexDataset(data_path, mean, std, is_augment)
            case 'real':
                return DualStreamRealDataset(data_path, mean, std, is_augment)
            case 'ap':
                return DualStreamAPDataset(data_path, mean, std, is_augment)
            case 'ap_db':
                return DualStreamAPDataset(data_path, mean, std, is_augment, is_db=True)
            case 'ap_c2_db':
                return DualStreamAPDatasetC2(data_path, mean, std, is_augment, is_db=True)
            case _:
                return DualStreamBaseDataset(data_path, mean, std, is_augment)
    else:
        match data_type:
            case 'complex':
                return ComplexDataset(data_path, mean, std, is_augment)
            case 'real':
                return RealDataset(data_path, mean, std, is_augment)
            case 'ap':
                return APDataset(data_path, mean, std, is_augment)
            case 'ap_db':
                return APDataset(data_path, mean, std, is_augment, is_db=True)
            case 'ap_c2_db':
                return APDatasetC2(data_path, mean, std, is_augment, is_db=True)
            case _:
                return BaseDataset(data_path, mean, std, is_augment)
