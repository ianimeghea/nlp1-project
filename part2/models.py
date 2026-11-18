import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class Classifier(nn.Module):
    def __init__(self, vectors, encoder="mean", hidden=128, rnn_hidden=128, bidirectional=False,
                 freeze=True, dropout=0.5, num_classes=7):
        super().__init__()
        self.encoder = encoder
        self.emb = nn.Embedding.from_pretrained(torch.tensor(vectors), freeze=freeze, padding_idx=0)
        size = vectors.shape[1]
        if encoder == "lstm":
            self.rnn = nn.LSTM(size, rnn_hidden, batch_first=True, bidirectional=bidirectional)
            size = rnn_hidden * (2 if bidirectional else 1)
        layers = [nn.Dropout(dropout)]
        if hidden:
            layers += [nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(dropout)]
            size = hidden
        layers.append(nn.Linear(size, num_classes))
        self.head = nn.Sequential(*layers)

    def encode(self, x, lengths):
        e = self.emb(x)
        mask = (x != 0).unsqueeze(-1)
        if self.encoder == "mean":
            return (e * mask).sum(1) / lengths.unsqueeze(1)
        if self.encoder == "max":
            return e.masked_fill(~mask, -1e9).max(1).values
        packed = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return torch.cat(list(h), dim=1)

    def forward(self, x, lengths):
        return self.head(self.encode(x, lengths))


def batches(X, y, idx, size):
    for s in range(0, len(idx), size):
        b = idx[s:s + size]
        seqs = [torch.tensor(X[i]) for i in b]
        lengths = torch.tensor([len(q) for q in seqs])
        yield pad_sequence(seqs, batch_first=True), lengths, torch.tensor(y[b])


def evaluate(model, X, y, idx):
    model.eval()
    total, probs = 0.0, []
    with torch.no_grad():
        for xb, lb, yb in batches(X, y, idx, 256):
            out = model(xb, lb)
            total += nn.functional.cross_entropy(out, yb, reduction="sum").item()
            probs.append(torch.softmax(out, -1).numpy())
    return total / len(idx), np.concatenate(probs)


def fit(X, y, train_idx, vectors, seed, lr=1e-3, weight_decay=1e-4, batch_size=32,
        max_epochs=30, patience=3, **kwargs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, dev = train_test_split(train_idx, test_size=0.1, stratify=y[train_idx], random_state=seed)
    model = Classifier(vectors, **kwargs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    best, state, bad, history = np.inf, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        for xb, lb, yb in batches(X, y, rng.permutation(tr), batch_size):
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb, lb), yb).backward()
            opt.step()
        train_loss, dev_loss = evaluate(model, X, y, tr)[0], evaluate(model, X, y, dev)[0]
        history.append([train_loss, dev_loss])
        if dev_loss < best:
            best, bad = dev_loss, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model, history
