import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data import LABELS, load_isear
from part2.embeddings import SOURCES, corpus_counts, load, lookup, lower, tokenize

OUT = Path("results/part2")
FIG = Path("figures/part2")
PAIRS = [("joy", "sadness"), ("happy", "sad"), ("love", "hate"), ("shame", "guilt"), ("anger", "disgust"), ("fear", "joy")]


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def coverage(texts, name, index):
    counts = corpus_counts(texts, lower(name))
    missing = Counter({w: c for w, c in counts.items() if lookup(w, index) is None})
    return {
        "types": len(counts),
        "type_coverage": 1 - len(missing) / len(counts),
        "token_coverage": 1 - sum(missing.values()) / sum(counts.values()),
        "top_missing": missing.most_common(15),
    }


def neighbours(word, words, vectors, index, k=8):
    sims = unit(vectors) @ unit(vectors[lookup(word, index)])
    return [words[i] for i in np.argsort(-sims)[1:k + 1]]


def zero_shot(texts, y, name, words, vectors, index):
    labels = unit(np.stack([vectors[lookup(w, index)] for w in LABELS]))
    correct = 0
    for text, label in zip(texts, y):
        rows = [r for r in (lookup(t, index) for t in tokenize(text, lower(name))) if r is not None]
        correct += int(np.argmax(labels @ unit(vectors[rows].mean(0))) == label)
    return correct / len(texts)


def analogy(a, b, c, words, vectors, index, k=5):
    v = vectors[lookup(b, index)] - vectors[lookup(a, index)] + vectors[lookup(c, index)]
    sims = unit(vectors) @ unit(v)
    return [words[i] for i in np.argsort(-sims) if words[i] not in (a, b, c)][:k]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    texts, y = load_isear()
    report = {}
    for name in SOURCES:
        words, vectors = load(name)
        index = {w: i for i, w in enumerate(words)}
        sim = lambda a, b: float(unit(vectors[lookup(a, index)]) @ unit(vectors[lookup(b, index)]))
        report[name] = {
            "coverage": coverage(texts, name, index),
            "neighbours": {w: neighbours(w, words, vectors, index) for w in LABELS},
            "pairs": {f"{a}-{b}": sim(a, b) for a, b in PAIRS},
            "zero_shot": zero_shot(texts, y, name, words, vectors, index),
            "analogy": {"man:doctor::woman:?": analogy("man", "doctor", "woman", words, vectors, index),
                        "woman:nurse::man:?": analogy("woman", "nurse", "man", words, vectors, index)},
        }
        print(name, {k: v for k, v in report[name]["coverage"].items() if k != "top_missing"},
              "zero-shot", round(report[name]["zero_shot"], 4))
        if name == "glove300":
            m = np.array([[sim(a, b) for b in LABELS] for a in LABELS])
            plt.figure(figsize=(5.5, 4.5))
            sns.heatmap(m, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=".2f", cmap="Blues", cbar=False)
            plt.title("GloVe-300 cosine similarity of the emotion labels")
            plt.tight_layout()
            plt.savefig(FIG / "label_similarity.png", dpi=200)
            plt.close()
    with open(OUT / "embeddings.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
