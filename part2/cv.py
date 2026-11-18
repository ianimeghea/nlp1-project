import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data import load_isear
from part2.embeddings import encode
from part2.models import evaluate, fit

POOL = dict(emb="glove300", encoder="mean", hidden=128, dropout=0.5, freeze=True, lr=1e-3)
LSTM = dict(POOL, encoder="lstm", rnn_hidden=128)

CONFIGS = {
    "pool": POOL,
    "pool-glove50": dict(POOL, emb="glove50"),
    "pool-glove100": dict(POOL, emb="glove100"),
    "pool-fasttext300": dict(POOL, emb="fasttext300"),
    "pool-max": dict(POOL, encoder="max"),
    "pool-linear": dict(POOL, hidden=0),
    "pool-nodropout": dict(POOL, dropout=0.0),
    "pool-lr3e-3": dict(POOL, lr=3e-3),
    "pool-trainable": dict(POOL, freeze=False),
    "pool-random": dict(POOL, freeze=False, random=True),
    "pool-data25": dict(POOL, fraction=0.25),
    "pool-data50": dict(POOL, fraction=0.5),
    "lstm": LSTM,
    "lstm-fasttext300": dict(LSTM, emb="fasttext300"),
    "lstm-bi": dict(LSTM, bidirectional=True),
    "lstm-h64": dict(LSTM, rnn_hidden=64),
    "lstm-h256": dict(LSTM, rnn_hidden=256),
    "lstm-nodropout": dict(LSTM, dropout=0.0),
    "lstm-trainable": dict(LSTM, freeze=False),
    "lstm-random": dict(LSTM, freeze=False, random=True),
    "lstm-data25": dict(LSTM, fraction=0.25),
    "lstm-data50": dict(LSTM, fraction=0.5),
    "pool-trainable-nodropout": dict(POOL, freeze=False, dropout=0.0),
    "pool-trainable-fasttext300": dict(POOL, freeze=False, emb="fasttext300"),
    "lstm-bi-h256": dict(LSTM, bidirectional=True, rnn_hidden=256),
    "lstm-bi-fasttext300": dict(LSTM, bidirectional=True, emb="fasttext300"),
}


def run_fold(texts, y, name, run, fold, train_idx, test_idx):
    torch.set_num_threads(1)
    config = dict(CONFIGS[name])
    seed = 1000 * run + fold
    X, vectors = encode(texts, config.pop("emb"), config.pop("random", False), seed)
    fraction = config.pop("fraction", 1.0)
    if fraction < 1:
        train_idx = np.random.default_rng(seed).choice(train_idx, int(fraction * len(train_idx)), replace=False)
    model, history = fit(X, y, train_idx, vectors, seed, **config)
    probs = evaluate(model, X, y, test_idx)[1]
    pred = probs.argmax(1)
    return {
        "config": name, "run": run, "fold": fold,
        "accuracy": accuracy_score(y[test_idx], pred),
        "macro_f1": f1_score(y[test_idx], pred, average="macro"),
        "epochs": len(history), "history": history,
        "pred": pred.tolist(), "confidence": probs.max(1).tolist(), "idx": test_idx.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/part2/final")
    args = p.parse_args()

    texts, labels = load_isear()
    y = np.array(labels)
    jobs = []
    for run in range(args.runs):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + run)
        for fold, (tr, te) in enumerate(skf.split(texts, y)):
            for name in args.configs:
                jobs.append((name, run, fold, tr, te))

    results = Parallel(n_jobs=-1, verbose=5)(delayed(run_fold)(texts, y, *job) for job in jobs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["config", "run", "fold", "accuracy", "macro_f1", "epochs"]
    scores = pd.DataFrame([{k: r[k] for k in keys} for r in results])
    scores.to_csv(out / "scores.csv", index=False)
    with open(out / "predictions.json", "w") as f:
        json.dump([r for r in results if r["run"] == 0], f)
    print(scores.groupby("config")[["accuracy", "macro_f1", "epochs"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
