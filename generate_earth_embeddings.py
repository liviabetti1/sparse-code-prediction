import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.location_encoder import load_location_encoder
from datasets import load_dataset, get_dataset_dir

embeddings_path_template = "{dataset}_{encoder}_{split}_embeddings.pt"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["satclip", "geoclip"], required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()

def load_or_generate(encoder_name, dataset_name, split, batch_size, device, output_dir=None):
    filename = embeddings_path_template.format(dataset=dataset_name, encoder=encoder_name, split=split)
    out_dir = output_dir or get_dataset_dir(dataset_name)
    path = os.path.join(out_dir, filename)

    if os.path.exists(path):
        print(f"Loading cached embeddings from {path}")
        data = torch.load(path)
        return data["embeddings"].numpy(), data["labels"].numpy()

    dataset = load_dataset(dataset_name, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    location_encoder = load_location_encoder(encoder_name, device=device)
    embeddings, labels = generate_earth_embeddings(location_encoder, dataloader, device)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels}, path)
    print(f"Saved embeddings to {path}")
    return embeddings.numpy(), labels.numpy()

def generate_earth_embeddings(location_encoder, dataloader, device):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating location embeddings..."):
            locations = batch["location"].to(device)
            labels = batch["label"].to(device)
            embeddings = location_encoder(locations)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_embeddings), torch.cat(all_labels)

def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    for split in args.splits:
        load_or_generate(
            encoder_name=args.encoder,
            dataset_name=args.dataset,
            split=split,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()
