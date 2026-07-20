import torch
from paths import load_paths

class _LocWrapper(torch.nn.Module):
    def __init__(self, enc): super().__init__(); self.enc = enc
    def encode_image(self, x): return self.enc(x)

def _splice_cfg():
    return load_paths()["splice"]

def load_splice_model(key: str, device: str = "cuda:0"):
    cfg = _splice_cfg()
    model_keys = [k for k in cfg if k != "root"]
    if key not in cfg:
        raise ValueError(f"Unknown splice model '{key}'. Available: {model_keys}")
    model = torch.load(cfg[key]["model"], map_location=device, weights_only=False)
    model.solver = "admm"
    model.device = device
    return model.to(device).eval()
