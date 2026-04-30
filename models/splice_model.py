import os
import yaml
import torch

_PATHS_YAML = os.path.join(os.path.dirname(__file__), "paths.yaml")


def _resolve_path(key: str) -> str:
    with open(_PATHS_YAML) as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("root", "")
    entry = cfg["splice"].get(key)
    if entry is None:
        raise ValueError(f"Unknown splice model key '{key}'. Available: {list(cfg['splice'].keys())}")
    return entry["model"].replace("${root}", root)


def load_splice_model(key: str, device: str = "cuda:0"):
    model_path = _resolve_path(key)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.solver = "admm"
    model.device = device
    return model.to(device).eval()
