import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets import get_dataset_dir
from models.splice_model import load_splice_model
from generate_earth_embeddings import load_or_generate as load_or_generate_earth_embeddings

embeddings_path_template = "{dataset}_splice_{model_name}_{encoder}_{split}_embeddings.pt"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splice_model", type=str, required=True)
    p.add_argument("--encoder", choices=["satclip", "geoclip"], required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()

def load_or_generate(splice_model_key, encoder_name, dataset_name, split, batch_size, device, output_dir=None):
    filename = embeddings_path_template.format(
        dataset=dataset_name, model_name=splice_model_key, encoder=encoder_name, split=split
    )
    out_dir = output_dir or get_dataset_dir(dataset_name)
    path = os.path.join(out_dir, filename)

    if os.path.exists(path):
        print(f"Loading cached embeddings from {path}")
        data = torch.load(path)
        return data["embeddings"].numpy(), data["labels"].numpy()

    loc_embeddings, labels = load_or_generate_earth_embeddings(
        encoder_name=encoder_name,
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        device=device,
        output_dir=output_dir,
    )
    loc_embeddings = torch.from_numpy(loc_embeddings)
    labels = torch.from_numpy(labels)

    dataloader = DataLoader(TensorDataset(loc_embeddings, labels), batch_size=batch_size, shuffle=False)
    splice_model = load_splice_model(splice_model_key, device=device)
    embeddings, labels = generate_splice_embeddings(splice_model, dataloader, device)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels}, path)
    print(f"Saved embeddings to {path}")
    return embeddings.numpy(), labels.numpy()

def generate_splice_embeddings(splice_model, dataloader, device):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for loc_embeddings, labels in tqdm(dataloader, desc="Generating splice embeddings..."):
            loc_embeddings = loc_embeddings.to(device)
            codes = splice_model.encode_image(loc_embeddings, return_cosine=False)
            all_embeddings.append(codes.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_embeddings), torch.cat(all_labels)

def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    for split in args.splits:
        load_or_generate(
            splice_model_key=args.splice_model,
            encoder_name=args.encoder,
            dataset_name=args.dataset,
            split=split,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()
