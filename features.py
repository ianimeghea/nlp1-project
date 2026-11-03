import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer


def token_counts(texts, tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    rows, cols = [], []
    for i, enc in enumerate(tokenizer.encode_batch(texts)):
        rows.extend([i] * len(enc.ids))
        cols.extend(enc.ids)
    data = np.ones(len(rows), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(len(texts), vocab_size))
    X.sum_duplicates()
    return X


class Indicator:
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        X.data[:] = 1.0
        return X


class BagOfWords:
    def fit(self, X):
        return self

    def transform(self, X):
        return X.copy()


class TfIdf:
    def fit(self, X):
        self.t = TfidfTransformer(sublinear_tf=True).fit(X)
        return self

    def transform(self, X):
        return self.t.transform(X).astype(np.float32)


FEATURES = {"indicator": Indicator, "bow": BagOfWords, "tfidf": TfIdf}
