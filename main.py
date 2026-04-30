import argparse
import os
import pickle

import numpy as np
import torch

from generate_earth_embeddings import load_or_generate as load_or_generate_earth_embeddings
from train import train_ridge
from eval import top_concepts


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["satclip", "geoclip"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=["ridge"], default="ridge")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--cv", action="store_true", help="Use RidgeCV to select alpha")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    X_train, y_train = load_or_generate_earth_embeddings(
        args.encoder, args.dataset, "train", args.batch_size, device, args.embeddings_dir
    )
    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(X_train.shape[0])
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    X_test, y_test = load_or_generate_earth_embeddings(
        args.encoder, args.dataset, "test", args.batch_size, device, args.embeddings_dir
    )

    print("Training ridge regression...")
    model = train_ridge(X_train, y_train, alpha=args.alpha, cv=args.cv)

    score = model.score(X_test, y_test)
    print(f"R² on test set: {score:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{args.dataset}_{args.encoder}_ridge.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
