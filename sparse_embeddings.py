import os
from typing import Optional
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, get_dataset_dir

embeddings_path_template = "{dataset}_sae_{model_tag}_{split}_embeddings.pt"

def _load_model(model_path: str, device: str):
    model = torch.load(model_path, map_location=device, weights_only=False)
    return model.to(device).eval()

def load_sparse_embeddings(
    model_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    batch_size: int = 512,
    device: str = "cuda:0",
    output_dir: Optional[str] = None,
    embeddings_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate SAE sparse embeddings.

    Provide either `embeddings_path` to load directly from a specific file,
    or `model_path` + `dataset_name` + `split` to derive the cache path automatically.
    """
    assert model_path is not None or embeddings_path is not None, \
        "Provide either model_path (to derive cache path) or embeddings_path (explicit path)"

    if embeddings_path is not None:
        path = embeddings_path
        out_dir = os.path.dirname(embeddings_path) or "."
    else:
        model_tag = os.path.splitext(os.path.basename(model_path))[0]
        filename = embeddings_path_template.format(dataset=dataset_name, model_tag=model_tag, split=split)
        out_dir = output_dir or get_dataset_dir(dataset_name)
        path = os.path.join(out_dir, filename)

    if os.path.exists(path):
        print(f"Loading cached embeddings from {path}")
        data = torch.load(path)
        return data["embeddings"].numpy(), data["labels"].numpy()

    dataset = load_dataset(dataset_name, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = _load_model(model_path, device)

    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating sparse embeddings..."):
            loc = batch["location"].to(device)
            codes = model(loc)
            all_embeddings.append(codes.cpu())
            all_labels.append(batch["label"].cpu())

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels}, path)
    print(f"Saved embeddings to {path}")
    return embeddings.numpy(), labels.numpy()
