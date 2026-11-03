import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class LogLinear:
    def __init__(self, C=1.0, seed=0):
        self.clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)


def mlp(input_dim, num_classes, hidden=(256,), dropout=0.5, activation=nn.ReLU):
    layers, d = [], input_dim
    for h in hidden:
        layers += [nn.Linear(d, h), activation(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, num_classes))
    return nn.Sequential(*layers)


class NeuralClassifier:
    def __init__(self, hidden=(256,), dropout=0.5, activation=nn.ReLU, lr=1e-3,
                 weight_decay=1e-4, batch_size=64, max_epochs=40, patience=4,
                 dev_fraction=0.1, seed=0):
        self.hidden = hidden
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.dev_fraction = dev_fraction
        self.seed = seed

    def _batches(self, X, y, shuffle, rng=None):
        idx = np.arange(X.shape[0])
        if shuffle:
            rng.shuffle(idx)
        for s in range(0, len(idx), self.batch_size):
            b = idx[s:s + self.batch_size]
            xb = torch.from_numpy(X[b].toarray())
            yb = None if y is None else torch.from_numpy(y[b])
            yield xb, yb

    def _loss(self, X, y):
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in self._batches(X, y, shuffle=False):
                total += nn.functional.cross_entropy(self.net(xb), yb, reduction="sum").item()
        return total / X.shape[0]

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, dtype=np.int64)
        tr, dev = train_test_split(np.arange(X.shape[0]), test_size=self.dev_fraction,
                                   stratify=y, random_state=self.seed)
        Xtr, ytr, Xdev, ydev = X[tr], y[tr], X[dev], y[dev]

        self.net = mlp(X.shape[1], int(y.max()) + 1, self.hidden, self.dropout, self.activation)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best, best_state, bad = np.inf, None, 0
        self.history = []
        for epoch in range(self.max_epochs):
            self.net.train()
            for xb, yb in self._batches(Xtr, ytr, shuffle=True, rng=rng):
                opt.zero_grad()
                nn.functional.cross_entropy(self.net(xb), yb).backward()
                opt.step()
            dev_loss = self._loss(Xdev, ydev)
            self.history.append(dev_loss)
            if dev_loss < best:
                best, best_state, bad = dev_loss, copy.deepcopy(self.net.state_dict()), 0
                self.best_epoch = epoch
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net.eval()
        out = []
        with torch.no_grad():
            for xb, _ in self._batches(X, None, shuffle=False):
                out.append(torch.softmax(self.net(xb), -1).numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


MODELS = {
    "loglin": lambda seed: LogLinear(seed=seed),
    "mlp": lambda seed: NeuralClassifier(hidden=(256,), dropout=0.5, seed=seed),
    "deep": lambda seed: NeuralClassifier(hidden=(512, 256), dropout=0.5,
                                          activation=nn.GELU, seed=seed),
}
