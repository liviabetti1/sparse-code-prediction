import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from external.satclip.satclip.load import get_satclip
from models.utils_loc import *

def load_location_encoder(location_model: str, device: str = "cuda:0"):
    """Load a pretrained location encoder."""
    return LocationEncoder(location_model=location_model, device=device)

class LocationEncoder(nn.Module):

    def __init__(
        self,
        location_model: str = None,
        device: str = "cuda:0"
    ):
        super().__init__()
        assert location_model is not None, "Must specify location model"
        self.location_model = location_model
        self.location_embedding_dim = LOCATION_EMBEDDING_DIMENSIONS[location_model]
        self.device = device

        self._load_location_model()
        #self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 2, "Forward expects (lat, lon) pairs"
        x = x[:, [1, 0]] if self.location_model in ["satclip", "gair", "climplicit"] else x
        x = x.double() if self.location_model in ["satclip"] else x.float()

        enc_device = next(self.location_encoder.parameters()).device
        if x.device != enc_device:
            x = x.to(enc_device, non_blocking=True)

        embedding = self.location_encoder(x).float()
        # embedding = F.normalize(embedding, dim=-1)

        # if self.normalize:
        #     embedding = F.normalize(embedding, dim=-1)

        return embedding

    def _load_location_model(self):
        """Load a pretrained location encoder."""
        if self.location_model == "satclip":
            from external.satclip.satclip.load import get_satclip
            from huggingface_hub import hf_hub_download
            self.location_encoder = get_satclip(
                hf_hub_download(LOCATION_MODEL_IDS["satclip"], LOCATION_MODEL_CHECKPOINTS["satclip"]),
                device=self.device,
            )
        
        elif self.location_model == "geoclip":
            from geoclip import GeoCLIP
            self.location_encoder = GeoCLIP().location_encoder
        
        elif self.location_model == "climplicit":
            from rshf.climplicit import Climplicit
            self.location_encoder = Climplicit.from_pretrained(LOCATION_MODEL_IDS["climplicit"], config={"return_chelsa": False})
        
        elif self.location_model == "gair":
            from huggingface_hub import hf_hub_download
            from external.GAIR.gair import GAIRModel
            checkpoint = hf_hub_download(repo_id=LOCATION_MODEL_IDS["gair"], filename=LOCATION_MODEL_CHECKPOINTS["gair"])
            model = GAIRModel.from_checkpoint(checkpoint, device=self.device, query_mode="nili")
            self.location_encoder = model.location_encoder

        elif self.location_model.startswith("csp"):
            self.location_encoder = load_csp(LOCATION_MODEL_CHECKPOINTS[location_model], device)

        elif self.location_model == "sinr":
            self.location_encoder = load_sinr(LOCATION_MODEL_CHECKPOINTS["sinr"], device)
        
        elif self.location_model == "taxabind":
            from transformers import PretrainedConfig
            from rshf.taxabind import TaxaBind
            config = PretrainedConfig.from_pretrained(LOCATION_MODEL_IDS['taxabind'])
            taxabind = TaxaBind(config)
            self.location_encoder =  taxabind.get_location_encoder()

        else:
            raise ValueError(f"Location model '{location_model}' is not supported")

