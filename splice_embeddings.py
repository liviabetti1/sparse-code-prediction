import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, get_dataset_dir
from models.splice_model import load_splice_model, _LocWrapper

embeddings_path_template = "{dataset}_splice_{model_name}_latlon_{split}_embeddings.pt"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splice_model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()

def load_or_generate(splice_model_key, dataset_name, split, batch_size=None, device=None, output_dir=None):
    filename = embeddings_path_template.format(
        dataset=dataset_name, model_name=splice_model_key, split=split
    )
    out_dir = output_dir or get_dataset_dir(dataset_name)
    path = os.path.join(out_dir, filename)

    if os.path.exists(path):
        print(f"Loading cached embeddings from {path}")
        data = torch.load(path)
        return data["embeddings"].numpy(), data["labels"].numpy()

    dataset = load_dataset(dataset_name, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    splice_model = load_splice_model(splice_model_key, device=device)
    splice_model.return_cosine = False
    embeddings, labels = generate_splice_embeddings(splice_model, dataloader, device)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels}, path)
    print(f"Saved embeddings to {path}")
    return embeddings.numpy(), labels.numpy()

def generate_splice_embeddings(splice_model, dataloader, device):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating splice embeddings..."):
            loc = batch["location"].to(device)
            codes = splice_model.encode_image(loc)
            all_embeddings.append(codes.cpu())
            all_labels.append(batch["label"].cpu())
    return torch.cat(all_embeddings), torch.cat(all_labels)

def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    for split in args.splits:
        load_or_generate(
            splice_model_key=args.splice_model,
            dataset_name=args.dataset,
            split=split,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()
