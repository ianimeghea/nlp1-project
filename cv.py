import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data import load_isear
from features import FEATURES, token_counts
from models import MODELS
from tokenizer import get_tokenizer


def run_fold(X, y, feature, model, run, fold, train_idx, test_idx):
    torch.set_num_threads(1)
    seed = 1000 * run + fold
    enc = FEATURES[feature]().fit(X[train_idx])
    clf = MODELS[model](seed).fit(enc.transform(X[train_idx]), y[train_idx])
    pred = clf.predict(enc.transform(X[test_idx]))
    return {
        "feature": feature, "model": model, "run": run, "fold": fold,
        "accuracy": accuracy_score(y[test_idx], pred),
        "macro_f1": f1_score(y[test_idx], pred, average="macro"),
        "pred": pred.tolist(), "idx": test_idx.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--vocab", type=int, default=8000)
    p.add_argument("--features", nargs="+", default=list(FEATURES))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--out", default="results/main")
    p.add_argument("--jobs", type=int, default=-1)
    args = p.parse_args()

    texts, labels = load_isear()
    y = np.array(labels)
    X = token_counts(texts, get_tokenizer(args.vocab))

    jobs = []
    for run in range(args.runs):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=run)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            for feature, model in product(args.features, args.models):
                jobs.append((feature, model, run, fold, tr, te))

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_fold)(X, y, *job) for job in jobs
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred", "idx")} for r in results])
    scores.to_csv(out / "scores.csv", index=False)
    with open(out / "predictions.json", "w") as f:
        json.dump([r for r in results if r["run"] == 0], f)
    print(scores.groupby(["feature", "model"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
