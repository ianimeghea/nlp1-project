import io
import zipfile
from collections import Counter
from pathlib import Path

import nltk
import numpy as np

from data import load_isear

DIR = Path("embeddings")
SOURCES = {
    "glove50": ("glove.6B.zip", "glove.6B.50d.txt", True),
    "glove100": ("glove.6B.zip", "glove.6B.100d.txt", True),
    "glove300": ("glove.6B.zip", "glove.6B.300d.txt", True),
    "fasttext300": ("wiki-news-300d-1M.vec.zip", "wiki-news-300d-1M.vec", False),
}
TOP = 100000
FASTTEXT_FIXES = {"n't": "not", "``": '"', "''": '"'}

nltk.download("punkt_tab", quiet=True)


def tokenize(text, lower):
    tokens = nltk.word_tokenize(text)
    if lower:
        return [t.lower() for t in tokens]
    return [FASTTEXT_FIXES.get(t, t) for t in tokens]


def corpus_counts(texts, lower):
    return Counter(t for text in texts for t in tokenize(text, lower))


def read_vectors(name, keep):
    archive, member, lower = SOURCES[name]
    words, vectors = [], []
    with zipfile.ZipFile(DIR / archive) as z, io.TextIOWrapper(z.open(member), encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue
            if i < TOP or parts[0] in keep:
                words.append(parts[0])
                vectors.append(np.array(parts[1:], dtype=np.float32))
    return words, np.stack(vectors)


def load(name):
    path = DIR / f"{name}.npz"
    if not path.exists():
        texts, _ = load_isear()
        keep = set(corpus_counts(texts, SOURCES[name][2]))
        words, vectors = read_vectors(name, keep)
        np.savez(path, words=np.array(words), vectors=vectors)
    data = np.load(path)
    return list(data["words"]), data["vectors"]


def lower(name):
    return SOURCES[name][2]


def lookup(token, index):
    if token in index:
        return index[token]
    return index.get(token.lower())


def encode(texts, name, random=False, seed=0):
    words, vectors = load(name)
    index = {w: i for i, w in enumerate(words)}
    tokens = [tokenize(t, lower(name)) for t in texts]
    rows = sorted({lookup(t, index) for seq in tokens for t in seq} - {None})
    new = {r: i + 2 for i, r in enumerate(rows)}
    matrix = np.zeros((len(rows) + 2, vectors.shape[1]), dtype=np.float32)
    matrix[2:] = vectors[rows]
    if random:
        rng = np.random.default_rng(seed)
        matrix[1:] = rng.normal(0, vectors.std(), matrix[1:].shape)
    X = [np.array([new.get(lookup(t, index), 1) for t in seq]) for seq in tokens]
    return X, matrix
