import json
from itertools import combinations
from pathlib import Path

import baycomp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data import LABELS, load_isear
from features import TfIdf, token_counts
from models import LogLinear
from tokenizer import get_tokenizer

ROPE = 0.01
ROPES = [0.005, 0.01, 0.02]
FIG = Path("figures")
RES = Path("results")
FEAT_NAMES = {"indicator": "indicator", "bow": "BoW", "tfidf": "TF-IDF"}
MODEL_NAMES = {"loglin": "log-linear", "mlp": "MLP-1", "deep": "MLP-2"}

sns.set_theme(style="whitegrid", font_scale=1.0)


def name(feature, model):
    return f"{MODEL_NAMES[model]} + {FEAT_NAMES[feature]}"


def load_scores(path=RES / "main" / "scores.csv"):
    df = pd.read_csv(path).sort_values(["run", "fold"])
    df["system"] = [name(f, m) for f, m in zip(df.feature, df.model)]
    return df


def posterior(a, b, runs):
    ct = baycomp.CorrelatedTTest(np.asarray(a), np.asarray(b), rope=0.0, runs=runs)
    return st.t(ct.df, loc=ct.mean, scale=np.sqrt(ct.var))


def region_probs(t, rope):
    return t.cdf(-rope), t.cdf(rope) - t.cdf(-rope), 1 - t.cdf(rope)


def compare(df, metric="accuracy"):
    runs = df.run.nunique()
    systems = {s: g[metric].to_numpy() for s, g in df.groupby("system", sort=False)}
    meta = df.drop_duplicates("system").set_index("system")[["feature", "model"]]
    rows = []
    for a, b in combinations(systems, 2):
        fa, ma = meta.loc[a]
        fb, mb = meta.loc[b]
        if fa != fb and ma != mb:
            kind = "other"
        else:
            kind = "same feature" if fa == fb else "same model"
        t = posterior(systems[a], systems[b], runs)
        row = {"A": a, "B": b, "kind": kind, "mean_diff": t.mean(), "std": t.std(),
               "hdi_low": t.interval(0.95)[0], "hdi_high": t.interval(0.95)[1]}
        for r in ROPES:
            pa, pt, pb = region_probs(t, r)
            row[f"P(A>B)@{r}"], row[f"P(tie)@{r}"], row[f"P(B>A)@{r}"] = pa, pt, pb
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(pa, pt, pb, threshold=0.95):
    if pa > threshold:
        return "A better"
    if pb > threshold:
        return "B better"
    if pt > threshold:
        return "equivalent"
    return "undecided"


def plot_scores(df):
    order = df.groupby("system").accuracy.mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, y="system", x="accuracy", order=order, color="#dde3ea",
                showfliers=False, ax=ax)
    sns.stripplot(data=df, y="system", x="accuracy", order=order, hue="run",
                  palette="deep", size=3.5, alpha=0.7, jitter=0.2, ax=ax)
    ax.set(ylabel="", xlabel="accuracy (10-fold CV x 3 repeats)")
    ax.legend(title="repeat", loc="lower right", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "scores.png", dpi=200)
    plt.close(fig)
    return list(order)


def plot_posteriors(df, pairs):
    runs = df.run.nunique()
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.2))
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
        t = posterior(df[df.system == a].accuracy, df[df.system == b].accuracy, runs)
        lo, hi = t.ppf(0.0005), t.ppf(0.9995)
        x = np.linspace(min(lo, -2.5 * ROPE), max(hi, 2.5 * ROPE), 600)
        ax.plot(x, t.pdf(x), color="#1f4e79")
        ax.fill_between(x, t.pdf(x), alpha=0.15, color="#1f4e79")
        ax.axvspan(-ROPE, ROPE, color="teal", alpha=0.2, label=f"ROPE ±{ROPE}")
        pa, pt, pb = region_probs(t, ROPE)
        ax.set_title(f"A = {a}\nB = {b}", fontsize=9)
        ax.set_xlabel("mean accuracy difference (B − A)")
        ax.set_yticks([])
        ax.text(0.02, 0.95, f"P(A>B) {pa:.2f}\nP(tie) {pt:.2f}\nP(B>A) {pb:.2f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "posteriors.png", dpi=200)
    plt.close(fig)


def plot_matrix(df, order):
    runs = df.run.nunique()
    n = len(order)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i != j:
                t = posterior(df[df.system == b].accuracy, df[df.system == a].accuracy, runs)
                M[i, j] = region_probs(t, ROPE)[2]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=order, yticklabels=order, annot_kws={"size": 7}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title(f"P(row better than column by > {ROPE})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "win_matrix.png", dpi=200)
    plt.close(fig)


def plot_data(texts, labels):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    counts = pd.Series([LABELS[l] for l in labels]).value_counts().reindex(LABELS)
    axes[0].bar(counts.index, counts.values, color="#1f4e79")
    axes[0].set_title("instances per label", fontsize=10)
    axes[0].tick_params(axis="x", rotation=40)
    tok = get_tokenizer()
    lengths = [len(e.ids) for e in tok.encode_batch(texts)]
    axes[1].hist(lengths, bins=40, color="#1f4e79")
    axes[1].set_title("BPE tokens per text", fontsize=10)
    axes[1].set_xlim(0, np.percentile(lengths, 99))
    fig.tight_layout()
    fig.savefig(FIG / "data.png", dpi=200)
    plt.close(fig)
    return lengths


def plot_confusion(feature, model):
    preds = json.load(open(RES / "main" / "predictions.json"))
    _, labels = load_isear()
    y = np.array(labels)
    gold, pred = [], []
    for r in preds:
        if r["feature"] == feature and r["model"] == model:
            gold.extend(y[r["idx"]])
            pred.extend(r["pred"])
    cm = confusion_matrix(gold, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=LABELS,
                yticklabels=LABELS, annot_kws={"size": 8}, ax=ax)
    ax.set(xlabel="predicted", ylabel="gold")
    ax.set_title(name(feature, model), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(cm, index=LABELS, columns=LABELS)


def plot_vocab():
    rows = []
    for d in sorted(RES.glob("vocab-*")) + [RES / "main"]:
        s = pd.read_csv(d / "scores.csv")
        s = s[s.feature == "tfidf"]
        v = 8000 if d.name == "main" else int(d.name.split("-")[1])
        for m, g in s.groupby("model"):
            if m in ("loglin", "mlp"):
                rows.append({"vocab": v, "model": MODEL_NAMES[m], "mean": g.accuracy.mean(),
                             "sem": g.accuracy.std() / np.sqrt(len(g))})
    v = pd.DataFrame(rows).sort_values("vocab")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for m, g in v.groupby("model"):
        ax.errorbar(g.vocab, g["mean"], yerr=2 * g["sem"], marker="o", capsize=3, label=f"{m} + TF-IDF")
    ax.set_xscale("log")
    ax.set_xticks(sorted(v.vocab.unique()))
    ax.set_xticklabels([f"{x // 1000}k" for x in sorted(v.vocab.unique())])
    ax.set(xlabel="BPE vocabulary size", ylabel="accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "vocab.png", dpi=200)
    plt.close(fig)
    return v


def top_features(k=6):
    texts, labels = load_isear()
    tok = get_tokenizer()
    X = token_counts(texts, tok)
    enc = TfIdf().fit(X)
    clf = LogLinear().fit(enc.transform(X), np.array(labels)).clf
    inv = {i: t for t, i in tok.get_vocab().items()}
    out = {}
    for c, lab in enumerate(LABELS):
        top = np.argsort(-clf.coef_[c])[:k]
        out[lab] = [inv[i].replace("▁", "") for i in top]
    return out


def main():
    FIG.mkdir(exist_ok=True)
    texts, labels, stats = load_isear(return_stats=True)
    lengths = plot_data(texts, labels)

    df = load_scores()
    summary = (df.groupby("system")[["accuracy", "macro_f1"]].agg(["mean", "std"])
               .sort_values(("accuracy", "mean"), ascending=False))
    order = plot_scores(df)
    best = order[0]
    meta = df.drop_duplicates("system").set_index("system")

    comp = compare(df)
    comp["verdict"] = [verdict(r[f"P(A>B)@{ROPE}"], r[f"P(tie)@{ROPE}"], r[f"P(B>A)@{ROPE}"])
                       for _, r in comp.iterrows()]
    comp.to_csv(RES / "bayes_comparisons.csv", index=False)

    best_f, best_m = meta.loc[best, ["feature", "model"]]
    pairs = [(name(best_f, "loglin"), name(best_f, "mlp")),
             (name(best_f, "mlp"), name(best_f, "deep")),
             (name("indicator", best_m), name("tfidf", best_m))]
    pairs = [p for p in pairs if p[0] != p[1]][:3]
    plot_posteriors(df, pairs)
    plot_matrix(df, order)
    cm = plot_confusion(best_f, best_m)
    vocab = plot_vocab()
    tops = top_features()

    report = {
        "data_stats": stats,
        "length_median": float(np.median(lengths)),
        "length_p95": float(np.percentile(lengths, 95)),
        "summary": summary.round(4).to_string(),
        "best": best,
        "posterior_pairs": pairs,
        "vocab": vocab.round(4).to_string(index=False),
        "top_features": tops,
        "confusion": cm.round(2).to_string(),
    }
    with open(RES / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(stats))
    print(summary.round(4))
    cols = ["A", "B", "kind", "mean_diff", f"P(A>B)@{ROPE}", f"P(tie)@{ROPE}", f"P(B>A)@{ROPE}", "verdict"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 30):
        print(comp[comp.kind != "other"][cols].round(3).to_string(index=False))
        print(comp[(comp.A == best) | (comp.B == best)][cols].round(3).to_string(index=False))
    print(vocab.round(4).to_string(index=False))
    print(tops)
    print(cm.round(2))


if __name__ == "__main__":
    main()
