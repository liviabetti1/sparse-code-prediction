import torch

class _LocWrapper(torch.nn.Module):
    def __init__(self, enc): super().__init__(); self.enc = enc
    def encode_image(self, x): return self.enc(x)
