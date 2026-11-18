import json
from itertools import combinations
from pathlib import Path

import baycomp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RES = Path("results/part2")
FIG = Path("figures/part2")
ROPE = 0.01
BASELINE = "log-linear + TF-IDF (Part 1)"

sns.set_theme(style="whitegrid")


def plot_search(search, part1):
    stats = search.groupby("config").accuracy.agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
    for ax, kind in zip(axes, ["pool", "lstm"]):
        s = stats[[c.split("-")[0] == kind for c in stats.index]].sort_values("mean")
        s = s[~s.index.str.contains("data")]
        colors = ["C3" if c == kind else "C0" for c in s.index]
        ax.barh(s.index, s["mean"], xerr=s["std"], color=colors, alpha=0.85)
        ax.axvline(part1, color="black", ls="--", lw=1, label=BASELINE)
        ax.set_title("embed-and-pool" if kind == "pool" else "LSTM")
        ax.set_xlim(0.3, 0.65)
        ax.set_xlabel("accuracy (5-fold search split)")
    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "search.png", dpi=200)
    plt.close()


def plot_data_size(search):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for kind, label in [("pool", "embed-and-pool"), ("lstm", "LSTM")]:
        names = [f"{kind}-data25", f"{kind}-data50", kind]
        g = search[search.config.isin(names)].groupby("config").accuracy
        ax.errorbar([25, 50, 100], g.mean()[names], yerr=g.std()[names], marker="o", capsize=3, label=label)
    ax.set_xlabel("% of training fold used")
    ax.set_ylabel("accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "data_size.png", dpi=200)
    plt.close()


def plot_overfitting(predictions, names):
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), sharey=True)
    for ax, name in zip(axes, names):
        history = np.array(next(r["history"] for r in predictions if r["config"] == name and r["fold"] == 0))
        ax.plot(range(1, len(history) + 1), history[:, 0], marker="o", label="train")
        ax.plot(range(1, len(history) + 1), history[:, 1], marker="o", label="dev")
        ax.axvline(np.argmin(history[:, 1]) + 1, color="grey", ls=":")
        ax.set_title(name)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(FIG / "overfitting.png", dpi=200)
    plt.close()


def compare(final):
    systems = {s: g.sort_values(["run", "fold"]).accuracy.to_numpy() for s, g in final.groupby("system")}
    rows = []
    for a, b in combinations(systems, 2):
        test = baycomp.CorrelatedTTest(systems[a], systems[b], rope=ROPE, runs=3)
        pa, pt, pb = test.probs()
        rows.append({"A": a, "B": b, "mean_diff": systems[a].mean() - systems[b].mean(),
                     "P(A>B)": pa, "P(tie)": pt, "P(B>A)": pb})
    return pd.DataFrame(rows)


def plot_final(final):
    order = final.groupby("system").accuracy.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.boxplot(data=final, y="system", x="accuracy", order=order, ax=ax, color="C0", width=0.5)
    sns.stripplot(data=final, y="system", x="accuracy", order=order, ax=ax, color="black", size=3, alpha=0.5)
    ax.set_ylabel("")
    ax.set_xlabel("accuracy (10-fold x 3 repeats)")
    plt.tight_layout()
    plt.savefig(FIG / "final.png", dpi=200)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    part1 = pd.read_csv("results/main/scores.csv")
    part1 = part1[(part1.feature == "tfidf") & (part1.model == "loglin")].assign(system=BASELINE)
    first, second = pd.read_csv(RES / "search" / "scores.csv"), pd.read_csv(RES / "search2" / "scores.csv")
    search = pd.concat([first[~first.config.isin(second.config)], second])
    final = pd.read_csv(RES / "final" / "scores.csv").assign(system=lambda d: d.config)
    final = pd.concat([final, part1])[["system", "run", "fold", "accuracy", "macro_f1"]]

    plot_search(search, part1.accuracy.mean())
    plot_data_size(search)
    plot_overfitting(json.load(open(RES / "search" / "predictions.json")), ["lstm", "lstm-trainable", "lstm-random"])
    plot_final(final)

    summary = final.groupby("system")[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4)
    comparisons = compare(final)
    print(search.groupby("config")[["accuracy", "epochs"]].agg(["mean", "std"]).round(4))
    print(summary)
    print(comparisons.round(3).to_string())
    summary.to_csv(RES / "final_summary.csv")
    comparisons.to_csv(RES / "comparisons.csv", index=False)


if __name__ == "__main__":
    main()
