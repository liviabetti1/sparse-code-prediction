import yaml
from pathlib import Path

from datasets.EuroSAT import EuroSAT
from datasets.MOSAIKS import MOSAIKSDataset

_paths = yaml.safe_load((Path(__file__).parent / "paths.yaml").read_text())

_MOSAIKS_DATASETS = {"elevation", "tree_cover", "nightlights", "population_density"}

def load_dataset(name, split="train", **kwargs):
    if name == "eurosat":
        return EuroSAT(root=_paths["eurosat"], split=split, **kwargs)
    elif name in _MOSAIKS_DATASETS:
        return MOSAIKSDataset(root=_paths["mosaiks"][name], split=split, label=name, **kwargs)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: eurosat, elevation, tree_cover, nightlights, population_density")

def get_dataset_dir(name):
    if name == "eurosat":
        return _paths["eurosat"]
    elif name in _MOSAIKS_DATASETS:
        return _paths["mosaiks"][name]
    else:
        raise ValueError(f"Unknown dataset '{name}'")
