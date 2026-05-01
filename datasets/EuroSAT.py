# EuroSAT is implemented in torchgeo.datasets.EuroSAT: https://torchgeo.readthedocs.io/en/latest/api/datasets/eurosat.html#eurosat
# We modify the implementation here to support returing locations
# All original code is from torchgeo

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from torchgeo.datasets.errors import DatasetNotFoundError, RGBBandsMissingError
from torchgeo.datasets.geo import NonGeoClassificationDataset
from torchgeo.datasets.utils import (
    Path,
    Sample,
    download_and_extract_archive,
    download_url,
    extract_archive,
    rasterio_loader,
)


class EuroSAT(NonGeoClassificationDataset):
    """EuroSAT dataset. Original code from torchgeo: https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/eurosat.py#L28-L280

    The `EuroSAT <https://github.com/phelber/EuroSAT>`__ dataset is based on Sentinel-2
    satellite images covering 13 spectral bands and consists of 10 target classes with
    a total of 27,000 labeled and geo-referenced images.

    Dataset format:

    * rasters are 13-channel GeoTiffs
    * labels are values in the range [0,9]

    Dataset classes:

    * Annual Crop
    * Forest
    * Herbaceous Vegetation
    * Highway
    * Industrial Buildings
    * Pasture
    * Permanent Crop
    * Residential Buildings
    * River
    * Sea & Lake

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following papers:

    * https://ieeexplore.ieee.org/document/8736785
    * https://ieeexplore.ieee.org/document/8519248
    """

    url = 'https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/'
    filename = 'EuroSATallBands.zip'
    md5 = '5ac12b3b2557aa56e1826e981e8e200e'

    # For some reason the class directories are actually nested in this directory
    base_dir = os.path.join(
        'ds', 'images', 'remote_sensing', 'otherDatasets', 'sentinel_2', 'tif'
    )

    splits = ('train', 'val', 'test')
    split_filenames: ClassVar[dict[str, str]] = {
        'train': 'eurosat-train.txt',
        'val': 'eurosat-val.txt',
        'test': 'eurosat-test.txt',
    }
    split_md5s: ClassVar[dict[str, str]] = {
        'train': '908f142e73d6acdf3f482c5e80d851b1',
        'val': '95de90f2aa998f70a3b2416bfe0687b4',
        'test': '7ae5ab94471417b6e315763121e67c5f',
    }

    all_band_names = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B09',
        'B10',
        'B11',
        'B12',
        'B8A',
    )

    rgb_bands = ('B04', 'B03', 'B02')

    BAND_SETS: ClassVar[dict[str, tuple[str, ...]]] = {
        'all': all_band_names,
        'rgb': rgb_bands,
    }

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        bands: Sequence[str] = BAND_SETS['all'],
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
        return_locations: bool = True,
    ) -> None:
        """Initialize a new EuroSAT dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bands: a sequence of band names to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionadded:: 0.3
           The *bands* parameter.
        """
        self._data_root = root
        self.root = root
        self.split = split
        # Avoid conflict between ImageFolder.transforms and our transforms
        self.tg_transforms = transforms
        self.download = download
        self.checksum = checksum

        assert self.split in {'train', 'val', 'test'}

        self._validate_bands(bands)
        self.bands = bands
        self.band_indices = Tensor(
            [self.all_band_names.index(b) for b in bands if b in self.all_band_names]
        ).long()

        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, self.split_filenames[split])) as f:
            for fn in f:
                valid_fns.add(fn.strip().replace('.jpg', '.tif'))

        def is_in_split(x: Path) -> bool:
            return os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transforms,
            loader=rasterio_loader,
            is_valid_file=is_in_split,
        )

        self._load_locations()
        self.return_locations = return_locations

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        path, label = self.samples[index] # for me to check for now
        image, label = self._load_image(index)

        image = torch.index_select(image, dim=0, index=self.band_indices).float()
        sample = {'image': image, 'label': label}

        if self.tg_transforms is not None:
            sample = self.tg_transforms(sample)

        if self.return_locations:
            row = self.locations.iloc[index]
            assert path == row['filepath'], f"Path mismatch: {path} vs {row['filepath']}" # for me to check for now
            sample['location'] = torch.tensor([row['lat'], row['lon']], dtype=torch.float)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check split file
        filename = os.path.join(self.root, self.split_filenames[self.split])
        if not os.path.isfile(filename):
            if self.download:
                download_url(
                    self.url + self.split_filenames[self.split],
                    self.root,
                    md5=self.split_md5s[self.split] if self.checksum else None,
                )
            else:
                raise DatasetNotFoundError(self)

        # Check image directory
        directory = os.path.join(self.root, self.base_dir)
        zipfile = os.path.join(self.root, self.filename)
        if os.path.isdir(directory):
            return
        elif os.path.isfile(zipfile):
            extract_archive(zipfile)
        elif self.download:
            print(f'Downloading and extracting {self.url + self.filename} to {self.root}')
            download_and_extract_archive(
                self.url + self.filename,
                self.root,
                md5=self.md5 if self.checksum else None,
            )
        else:
            raise DatasetNotFoundError(self)

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided sequence of bands to load

        Raises:
            AssertionError: if ``bands`` is not a sequence
            ValueError: if an invalid band name is provided

        .. versionadded:: 0.3
        """
        assert isinstance(bands, Sequence), "'bands' must be a sequence"
        for band in bands:
            if band not in self.all_band_names:
                raise ValueError(f"'{band}' is an invalid band name.")


    def _load_locations(self) -> None:
        """Extract lat lon locations from the GeoTIFF metadata and save them in csv"""
        import rasterio
        import pyproj
        import pandas as pd
        from tqdm import tqdm

        cache = os.path.join(self._data_root, f'eurosat-{self.split}-locations.csv')
        if os.path.exists(cache):
            print(f'Loading cached locations from {cache}')
            self.locations = pd.read_csv(cache)
            return
        
        locations = []
        for filepath, _ in tqdm(self.samples, desc=f'Extracting locations for {self.split} split'):
            with rasterio.open(filepath) as src:
                centroid_x = (src.bounds.left + src.bounds.right) / 2
                centroid_y = (src.bounds.bottom + src.bounds.top) / 2
                transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True) # not sure if they all have the same crs?
                lon, lat = transformer.transform(centroid_x, centroid_y)
                locations.append((filepath, lon, lat))

        self.locations = pd.DataFrame(locations, columns=['filepath', 'lon', 'lat'])
        self.locations.to_csv(cache, index=False)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = np.take(sample['image'].numpy(), indices=rgb_indices, axis=0)
        image = np.rollaxis(image, 0, 3)
        image = np.clip(image / 3000, 0, 1)

        label = cast(int, sample['label'].item())
        label_class = self.classes[label]

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = cast(int, sample['prediction'].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')
        if show_titles:
            title = f'Label: {label_class}'
            if showing_predictions:
                title += f'\nPrediction: {prediction_class}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--root', type=str, default='/data/eurosat')
    args = parser.parse_args()
    # Specify all splits so it downloads each txt file
    # Note: downloading already unzips!
    _ = EuroSAT(root=args.root, split="train", download=args.download)
    _ = EuroSAT(root=args.root, split="val", download=args.download)
    _ = EuroSAT(root=args.root, split="test", download=args.download)