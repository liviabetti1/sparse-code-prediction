import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from external.satclip.satclip.load import get_satclip

LOCATION_EMBEDDING_DIMENSIONS = {
    "geoclip": 512,
    "satclip": 256,
}

LOCATION_MODEL_IDS = {
    "satclip": "microsoft/SatCLIP-ViT16-L40",
    # might need to include L10 as well, or can also look at ResNet-based models
}

LOCATION_MODEL_CHECKPOINTS = {
    "satclip": "satclip-vit16-l40.ckpt",
}

def load_location_encoder(location_model: str, device: str = "cuda:0"):
    """Load a pretrained location encoder."""
    return LocationEncoder(location_model=location_model, device=device)

class LocationEncoder(nn.Module):

    def __init__(
        self,
        location_model: str = None,
        normalize: bool = True,
        device: str = "cuda:0"
    ):
        super().__init__()
        assert location_model is not None, "Must specify location model"
        self.location_model = location_model
        self.location_embedding_dim = LOCATION_EMBEDDING_DIMENSIONS[location_model]
        self.device = device

        self._load_location_model()
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 2, "Forward expects (lat, lon) pairs"
        x = x[:, [1, 0]].double() if self.location_model == "satclip" else x.float() 

        enc_device = next(self.location_encoder.parameters()).device
        if x.device != enc_device:
            x = x.to(enc_device, non_blocking=True)

        embedding = self.location_encoder(x).float()

        if self.normalize:
            embedding = F.normalize(embedding, dim=-1)

        return embedding

    def _load_location_model(self):
        """Load a pretrained location encoder."""
        if self.location_model == "satclip":
            self.location_encoder = get_satclip(
                hf_hub_download(LOCATION_MODEL_IDS["satclip"], LOCATION_MODEL_CHECKPOINTS["satclip"]),
                device=self.device,
            )
        elif self.location_model == "geoclip":
            from geoclip import GeoCLIP
            self.location_encoder = GeoCLIP().location_encoder
        else:
            raise ValueError(f"Location model '{self.location_model}' is not supported")

