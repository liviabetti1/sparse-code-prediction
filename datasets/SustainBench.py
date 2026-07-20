"""SustainBench dataset wrappers for FMoW and DHS tasks.
Used code from sustainbench repo.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'sustainbench'))

from typing import Callable

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from sustainbench.datasets.fmow_dataset import FMoWDataset, categories as FMOW_CATEGORIES
from sustainbench.datasets.dhs_dataset import DHSDataset, BAND_ORDER as DHS_BAND_ORDER


class SustainBenchFMoW(Dataset):
    """SustainBench FMoW (Functional Map of the World) dataset.

    224x224 RGB satellite images across 62 land use / building categories.

    Returns samples as {'image': Tensor[3,224,224], 'label': LongTensor,
                        'location': Tensor[2]} where location is [lat, lon].

    Splits: 'train', 'val' (ID val by default), 'test' (OOD test).
    """

    classes = FMOW_CATEGORIES

    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        transforms: Callable | None = None,
        download: bool = False,
        return_locations: bool = True,
    ) -> None:
        self.split = split
        self.tg_transforms = transforms
        self.return_locations = return_locations

        self._base = FMoWDataset(root_dir=root, download=download)
        assert split in self._base.split_dict, (
            f"split must be one of {list(self._base.split_dict.keys())}"
        )
        self._subset = self._base.get_subset(split)
        self._indices = self._subset.indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        actual_idx = self._indices[index]
        img_pil = self._base.get_input(actual_idx)
        image = TF.to_tensor(img_pil)
        label = self._base._y_array[actual_idx]

        sample = {'image': image, 'label': label}

        if self.tg_transforms is not None:
            sample = self.tg_transforms(sample)

        if self.return_locations:
            # full_idxs maps from index space to raw metadata rows
            metadata_row = self._base.metadata.iloc[self._base.full_idxs[actual_idx]]
            sample['location'] = torch.tensor(
                [metadata_row['lat'], metadata_row['lon']], dtype=torch.float
            )

        return sample


class SustainBenchDHS(Dataset):
    """SustainBench DHS (Demographic and Health Surveys) dataset.

    224x224x8 Landsat + nightlights imagery, predicting a continuous asset
    wealth index (regression task).

    Returns samples as {'image': Tensor[8,H,W], 'label': FloatTensor,
                        'location': Tensor[2]} where location is [lat, lon].

    Bands: BLUE, GREEN, RED, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS.
    Splits: 'train', 'val' (OOD val by default), 'test' (OOD test).
    """

    band_names = DHS_BAND_ORDER

    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        transforms: Callable | None = None,
        download: bool = False,
        return_locations: bool = True,
        fold: str = 'A',
    ) -> None:
        self.split = split
        self.tg_transforms = transforms
        self.return_locations = return_locations

        self._base = DHSDataset(root_dir=root, download=download, fold=fold)
        assert split in self._base.split_dict, (
            f"split must be one of {list(self._base.split_dict.keys())}"
        )
        self._subset = self._base.get_subset(split)
        self._indices = self._subset.indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        actual_idx = self._indices[index]
        image = self._base.get_input(actual_idx)
        label = self._base._y_array[actual_idx]

        sample = {'image': image, 'label': label}

        if self.tg_transforms is not None:
            sample = self.tg_transforms(sample)

        if self.return_locations:
            row = self._base.metadata.iloc[actual_idx]
            sample['location'] = torch.tensor(
                [row['lat'], row['lon']], dtype=torch.float
            )

        return sample


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['fmow', 'dhs'], required=True)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()

    cls = SustainBenchFMoW if args.dataset == 'fmow' else SustainBenchDHS
    for split in ('train', 'val', 'test'):
        ds = cls(root=args.root, split=split, download=args.download)
        print(f'{split}: {len(ds)} samples')
        sample = ds[0]
        print(f"  image: {sample['image'].shape}, label: {sample['label']}, location: {sample['location']}")
