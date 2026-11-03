import re
from collections import defaultdict

from datasets import load_dataset

LABELS = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

PLACEHOLDER = re.compile(
    r"^(no response|not applicable|does not apply|doesn'?t apply|nothing|none|"
    r"can'?t think of any|can'?t remember|i can'?t remember|blank)\b",
    re.I,
)


def clean(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.strip("[] ").strip()


def is_placeholder(text):
    return len(text.split()) <= 6 and bool(PLACEHOLDER.match(text))


def load_isear(return_stats=False):
    raw = load_dataset("gsri-18/ISEAR-dataset-complete")["train"]
    stats = {"raw": len(raw), "empty_or_placeholder": 0, "duplicates": 0, "conflicting": 0}

    first_seen = {}
    labels_seen = defaultdict(set)
    for text, label in zip(raw["content"], raw["emotion"]):
        text = clean(text)
        if not text or is_placeholder(text):
            stats["empty_or_placeholder"] += 1
            continue
        key = text.lower()
        if key in first_seen:
            stats["duplicates"] += 1
        else:
            first_seen[key] = text
        labels_seen[key].add(label)

    texts, labels = [], []
    for key, text in first_seen.items():
        if len(labels_seen[key]) > 1:
            stats["conflicting"] += 1
            continue
        texts.append(text)
        labels.append(LABELS.index(next(iter(labels_seen[key]))))

    stats["kept"] = len(texts)
    if return_stats:
        return texts, labels, stats
    return texts, labels


def load_tokenizer_corpus():
    return load_dataset("dair-ai/emotion", "unsplit")["train"]["text"]
