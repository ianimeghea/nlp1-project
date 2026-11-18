import json
import re
from pathlib import Path

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from data import LABELS, load_isear
from part2.embeddings import load, lookup, tokenize

OUT = Path("results/part2")
FIG = Path("figures/part2")
NEGATION = {"not", "n't", "no", "never", "nothing", "nobody", "without"}
EMOTION_WORDS = {
    "joy": r"\b(joy\w*|happ\w*|glad|delight\w*)",
    "fear": r"\b(fear\w*|afraid|scar\w*|frighten\w*|terrif\w*)",
    "anger": r"\b(ang\w*|furious|annoy\w*|irritat\w*)",
    "sadness": r"\b(sad\w*|unhapp\w*|depress\w*|cry\w*|cried)",
    "disgust": r"\b(disgust\w*|sick\w*|revolt\w*)",
    "shame": r"\b(sham\w*|asham\w*|embarrass\w*)",
    "guilt": r"\b(guilt\w*)",
}


def features(texts, y):
    words, _ = load("glove300")
    index = {w: i for i, w in enumerate(words)}
    rows = []
    for text, label in zip(texts, y):
        tokens = tokenize(text, True)
        rows.append({
            "log_length": np.log(len(tokens)),
            "oov_rate": np.mean([lookup(t, index) is None for t in tokens]),
            "negation": float(any(t in NEGATION for t in tokens)),
            "names_emotion": float(bool(re.search(EMOTION_WORDS[LABELS[label]], text.lower()))),
        })
    return pd.DataFrame(rows)


def correctness(path, keep, y):
    rows = {}
    for r in json.load(open(path)):
        name = r.get("config") or f"{r['model']}-{r['feature']}"
        if name in keep:
            pred = rows.setdefault(keep[name], np.zeros(len(y), dtype=int))
            pred[r["idx"]] = np.array(r["pred"]) == y[r["idx"]]
    return rows


def model(system, x, num_systems, correct=None):
    bias = numpyro.sample("bias", dist.Normal(0, 2))
    with numpyro.plate("systems", num_systems):
        shift = numpyro.sample("shift", dist.Laplace(0, 1))
        w = numpyro.sample("w", dist.Laplace(0, 1).expand([x.shape[1]]).to_event(1))
    logit = bias + shift[system] + (w[system] * x).sum(-1)
    numpyro.sample("correct", dist.Bernoulli(logits=logit), obs=correct)


def main(pool="pool-trainable", lstm="lstm-bi-fasttext300"):
    texts, labels = load_isear()
    y = np.array(labels)
    f = features(texts, y)
    systems = correctness("results/main/predictions.json", {"loglin-tfidf": "log-linear + TF-IDF"}, y)
    systems.update(correctness("results/part2/final/predictions.json", {pool: "embed-and-pool (fine-tuned)", lstm: "BiLSTM (fastText)"}, y))
    names = list(systems)

    summary = f.copy()
    for s in names:
        summary[s] = systems[s]
    for col in ["negation", "names_emotion"]:
        print(summary.groupby(col)[names].mean().round(3))
    summary["length_bin"] = pd.qcut(np.exp(summary.log_length), 4)
    print(summary.groupby("length_bin", observed=True)[names].mean().round(3))

    x = (f.to_numpy() - f.to_numpy().mean(0)) / f.to_numpy().std(0)
    system = np.repeat(np.arange(len(names)), len(y))
    correct = np.concatenate([systems[s] for s in names])
    mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000, num_chains=2, chain_method="sequential")
    mcmc.run(jax.random.PRNGKey(0), system, np.tile(x, (len(names), 1)), len(names), correct)
    post = az.from_numpyro(mcmc, coords={"system": names, "feature": list(f.columns)},
                           dims={"shift": ["system"], "w": ["system", "feature"]})
    table = az.summary(post, var_names=["w"], ci_prob=0.95)
    print(table)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "error_analysis.csv")

    w = post.posterior["w"].stack(sample=("chain", "draw")).values
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, s in enumerate(names):
        for j, feat in enumerate(f.columns):
            lo, hi = np.percentile(w[i, j], [2.5, 97.5])
            pos = j + (i - 1) * 0.22
            ax.plot([lo, hi], [pos, pos], color=f"C{i}", lw=2)
            ax.plot(w[i, j].mean(), pos, "o", color=f"C{i}", label=s if j == 0 else None)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(f.columns)), [c.replace("_", " ") for c in f.columns])
    ax.set_xlabel("effect on log-odds of a correct prediction (per std)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG / "error_analysis.png", dpi=200)


if __name__ == "__main__":
    main()
