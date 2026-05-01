import os
import yaml
import torch

_PATHS_YAML = os.path.join(os.path.dirname(__file__), "..", "paths.yaml")

class _LocWrapper(torch.nn.Module):
    def __init__(self, enc): super().__init__(); self.enc = enc
    def encode_image(self, x): return self.enc(x)

def _splice_cfg():
    with open(_PATHS_YAML) as f:
        cfg = yaml.safe_load(f)["splice"]
    return cfg["root"], cfg["models"]

def load_splice_model(key: str, device: str = "cuda:0"):
    root, models = _splice_cfg()
    if key not in models:
        raise ValueError(f"Unknown splice model '{key}'. Available: {list(models.keys())}")
    model = torch.load(os.path.join(root, models[key]["model"]), map_location=device, weights_only=False)
    model.solver = "admm"
    model.device = device
    return model.to(device).eval()
