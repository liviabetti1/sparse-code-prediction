import argparse, os, pickle
import torch
from sklearn.metrics import r2_score, accuracy_score
from earth_embeddings import load_or_generate as load_location_embeddings
from splice_embeddings import load_or_generate as load_splice_embeddings
from sparse_embeddings import load_sparse_embeddings
from train import train
from eval import top_concepts

from models.splice_model import _splice_cfg



def get_loader(args, device):
    if args.embeddings == "location":
        assert args.encoder, "--encoder required for location embeddings"
        return lambda split: load_location_embeddings(args.encoder, args.dataset, split, args.batch_size, device, args.embeddings_dir)
    elif args.embeddings == "sae":
        assert args.sae_model_path, "--sae_model_path required for sae embeddings"
        return lambda split: load_sparse_embeddings(
                model_path=args.sae_model_path,
                dataset_name=args.dataset,
                split=split,
                batch_size=args.batch_size,
                device=device,
                output_dir=args.embeddings_dir,
            )
    elif args.embeddings == "splice":
        assert args.splice_model, "--splice_model required for splice embeddings"
        return lambda split: load_splice_embeddings(args.splice_model, args.dataset, split, args.batch_size, device, args.embeddings_dir)
    else:
        raise NotImplementedError(f"Embedding type {args.embeddings} not yet implemented")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--embeddings", choices=["location", "splice", "sae"], default="location")
    p.add_argument("--task", choices=['regression', "classification"], default="regression")
    p.add_argument("--model", required=True, choices=["ridge_regression", "lasso_regression", "ridge_classifier", "logistic_regression"])

    # Location embeddings
    p.add_argument("--encoder", choices=["satclip", "geoclip", "gair", "climplicit"])
    # OR SpLiCE model name
    p.add_argument("--splice_model")
    # OR SAE model path (can also reformat to name)
    p.add_argument("--sae_model_path")

    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--cv", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--embeddings_dir", default=None)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



def score(model, X_test, y_test, task: str):
    y_pred = model.predict(X_test)
    if task == "regression":
        metric = r2_score(y_test, y_pred)
        print(f"R2 on test set: {metric:.4f}")
    else:
        metric = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test set: {metric:.4f}")
    return metric


def save(model, args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.embeddings == "location":
        tag = f"location_encoder_{args.encoder}"
    elif args.embeddings == "splice":
        tag = f"splice_{args.splice_model}"
    else:
        tag = f"sae_{os.path.splitext(os.path.basename(args.sae_model_path))[0]}"
    path = os.path.join(args.output_dir, f"{args.dataset}_{tag}_{args.model}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    load = get_loader(args, device)

    X_train, y_train = load("train")
    X_test, y_test = load("test")

    model = train(args.model, X_train, y_train, alpha=args.alpha, cv=args.cv)
    score(model, X_test, y_test, args.task)

    if args.embeddings == "splice":
        root, models = _splice_cfg()
        concepts = torch.load(os.path.join(root, models[args.splice_model]["concepts"]))
        tc = top_concepts(model, k=args.topk, concepts=concepts)
        if args.task == "classification":
            tc = top_concepts(model, k=args.topk, concepts=concepts, abs_val=False) # put abs val as False for now to see what concepts are contributing to class
            assert isinstance(tc, dict), "Error in top concepts for classification"
            for cls, concepts_ in tc.items():
                print(f"Class {cls} top {args.topk} concepts: {concepts_}")
        else:
            tc = top_concepts(model, k=args.topk, concepts=concepts)
            print(f"Top {args.topk} concepts: {tc}")

    save(model, args)


if __name__ == "__main__":
    main()
