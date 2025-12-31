# data_pipeline.py
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from collections import Counter

PAD, UNK = "<PAD>", "<UNK>"


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def load_splits(remove_parts, seed, val_split):
    data = fetch_20newsgroups(subset="train", remove=remove_parts)
    X, y = data.data, data.target
    target_names = data.target_names
    num_classes = len(target_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    test = fetch_20newsgroups(subset="test", remove=remove_parts)
    return X_train, X_val, test.data, y_train, y_val, test.target, target_names, num_classes


def build_vocab(X_train, min_freq, max_vocab):
    counter = Counter()
    for t in X_train:
        counter.update(tokenize(t))

    vocab_tokens = []
    for w, c in counter.items():
        if c >= min_freq:
            vocab_tokens.append(w)

    vocab_tokens.sort(key=lambda w: counter[w], reverse=True)
    vocab_tokens = vocab_tokens[: max_vocab - 2]

    itos = [PAD, UNK] + vocab_tokens
    stoi = {w: i for i, w in enumerate(itos)}

    return stoi, itos, stoi[PAD], stoi[UNK], len(stoi)


def load_glove(path, dim):
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return glove


def build_embedding_matrix(stoi, pad_id, glove, dim):
    vocab_size = len(stoi)
    matrix = np.random.normal(scale=0.02, size=(vocab_size, dim)).astype(np.float32)
    matrix[pad_id] = 0.0

    hits = 0
    for w, i in stoi.items():
        if w in (PAD, UNK):
            continue
        if w in glove:
            matrix[i] = glove[w]
            hits += 1

    return matrix, hits / max(1, vocab_size - 2)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, stoi, unk_id, max_len):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.unk_id = unk_id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])[: self.max_len]
        if not tokens:
            tokens = [UNK]
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        return torch.tensor(ids), torch.tensor(self.labels[idx])




def collate_fn(pad_id):
    def fn(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs])
        max_len = lengths.max()
        padded = torch.full((len(seqs), max_len), pad_id)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s
        return padded, torch.tensor(labels), lengths
    return fn


def make_loaders(Xtr, ytr, Xv, yv, Xt, yt, stoi, unk_id, pad_id, max_len, batch):
    train_loader = DataLoader(
        NewsDataset(Xtr, ytr, stoi, unk_id, max_len),
        batch_size=batch,
        shuffle=True,
        collate_fn=collate_fn(pad_id)
    )

    val_loader = DataLoader(
        NewsDataset(Xv, yv, stoi, unk_id, max_len),
        batch_size=batch,
        shuffle=False,
        collate_fn=collate_fn(pad_id)
    )

    test_loader = DataLoader(
        NewsDataset(Xt, yt, stoi, unk_id, max_len),
        batch_size=batch,
        shuffle=False,
        collate_fn=collate_fn(pad_id)
    )

    return train_loader, val_loader, test_loader
