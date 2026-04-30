"""MOSAIKS datasets

Includes:
    - Elevation
    - Tree cover
    - Nightlights
    - Population Density"""

import os
from typing import ClassVar, Literal

import torch
from torch.utils.data import Dataset 
import pandas as pd
import numpy as np

from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.utils import (
    Path,
    Sample
)


class MOSAIKSDataset(Dataset):
    """
    """

    splits = ('train', 'val', 'test')
    split_filenames: ClassVar[dict[str, str]] = {
        'all': 'data.csv',
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }
    labels = ('elevation', 'tree_cover', 'nightlights', 'population_density')
    label_cols = {
        'elevation': 'elevation',
        'tree_cover': 'treecover',
        'nightlights': 'luminosity',
        'population_density': 'population' 
    }
    logged_values = {
        'elevation': False,
        'tree_cover': False,
        'nightlights': True,
        'population_density': True
    }

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        label: str | None = None,
        download: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.label = label
        self.download = download

        assert self.split in {'train', 'val', 'test'}
        assert self.label in {'elevation', 'tree_cover', 'nightlights', 'population_density'}
        self.label_col = self.label_cols[self.label]
        self.log_transform = self.logged_values[self.label]

        self._verify()

        if self.log_transform:
            self.df[self.label_col] = np.log1p(self.df[self.label_col])

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        row = self.df.iloc[index]
        lat = row['lat']
        lon = row['lon']
        value = row[f'{self.label_col}']

        return {
            'location': torch.tensor([lon, lat], dtype=torch.float),
            'label': torch.tensor(value, dtype=torch.float),
        }
    
    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.df)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.
        Used from torchgeo..."""
        # Check split file
        split_filename = os.path.join(self.root, self.split_filenames[self.split])
        all_data_filename = os.path.join(self.root, self.split_filenames['all'])
        if not os.path.isfile(split_filename):
            if not os.path.isfile(all_data_filename):
                if self.download:
                    raise NotImplementedError("Downloading not implemented yet for MOSAIKS dataset")
                else:
                    raise DatasetNotFoundError(self)
            
            all_data = pd.read_csv(all_data_filename)
            self._train_test_split(all_data)

        self.df = pd.read_csv(split_filename)

    def _train_test_split(self, all_data: pd.DataFrame, test_size: float = 0.1, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into train and test sets."""
        from sklearn.model_selection import train_test_split

        all_data = all_data[all_data.notna().all(axis=1)]

        train_df, test_df = train_test_split(all_data, test_size=test_size, random_state=random_state)
        train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=random_state)

        train_df.to_csv(os.path.join(self.root, self.split_filenames['train']), index=False)
        val_df.to_csv(os.path.join(self.root, self.split_filenames['val']), index=False)
        test_df.to_csv(os.path.join(self.root, self.split_filenames['test']), index=False)
    