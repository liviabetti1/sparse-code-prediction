import os
from pathlib import Path
from paths import load_paths

from datasets.EuroSAT import EuroSAT
from datasets.MOSAIKS import MOSAIKSDataset
from datasets.SustainBench import SustainBenchDHS

_p = load_paths()
_data_cfg = _p["data"]
_pred_ds = _p["prediction"]["datasets"]

# prediction.yaml uses "forest"/"population"; MOSAIKS.py uses "tree_cover"/"population_density"
_MOSAIKS_KEY_MAP = {
    "elevation":         "elevation",
    "tree_cover":        "forest",
    "nightlights":       "nightlights",
    "population_density": "population",
}
_MOSAIKS_DATASETS = set(_MOSAIKS_KEY_MAP.keys())


def _mosaiks_root(name):
    return os.path.dirname(_pred_ds["mosaiks"][_MOSAIKS_KEY_MAP[name]]["csv"])


def load_dataset(name, split="train", **kwargs):
    if name == "eurosat":
        return EuroSAT(root=str(Path(_data_cfg["root"]) / "eurosat"), split=split, **kwargs)
    elif name in _MOSAIKS_DATASETS:
        return MOSAIKSDataset(root=_mosaiks_root(name), split=split, label=name, **kwargs)
    elif name == "sustainbench":
        root = str(Path(_pred_ds["sustainbench"]["trainval"]).parent)
        return SustainBenchDHS(root=root, split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: eurosat, {', '.join(_MOSAIKS_DATASETS)}, sustainbench")


def get_dataset_dir(name):
    if name == "eurosat":
        return str(Path(_data_cfg["root"]) / "eurosat")
    elif name in _MOSAIKS_DATASETS:
        return _mosaiks_root(name)
    elif name == "sustainbench":
        return str(Path(_pred_ds["sustainbench"]["trainval"]).parent)
    else:
        raise ValueError(f"Unknown dataset '{name}'.")
