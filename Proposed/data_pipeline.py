import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from collections import Counter

PAD, UNK = "<PAD>", "<UNK>"

_word_re = re.compile(r"[a-z0-9']+")
_sent_re = re.compile(r"(?<=[.!?])\s+")

def tokenize_words(text: str):
    return _word_re.findall(text.lower())

def split_sentences(text: str):
    sents = _sent_re.split(text.strip())
    return [s for s in sents if s and s.strip()]

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
    for doc in X_train:
        counter.update(tokenize_words(doc))
    vocab_tokens = [w for w, c in counter.items() if c >= min_freq]
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


class NewsSentenceDataset(Dataset):
    def __init__(self, texts, labels, stoi, unk_id, max_sents, max_tokens_per_sent):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.unk_id = unk_id
        self.max_sents = max_sents
        self.max_tokens_per_sent = max_tokens_per_sent

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        doc = self.texts[idx]
        sents = split_sentences(doc)

        if not sents:
            sents = [doc]

        sents = sents[: self.max_sents]
        sent_ids = []
        sent_lens = []

        for s in sents:
            toks = tokenize_words(s)[: self.max_tokens_per_sent]
            if not toks:
                toks = [UNK]
            ids = [self.stoi.get(t, self.unk_id) for t in toks]
            sent_ids.append(torch.tensor(ids, dtype=torch.long))
            sent_lens.append(len(ids))

        return sent_ids, torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(sent_lens, dtype=torch.long)


def collate_sentence_batch(pad_id, max_sents, max_tokens_per_sent):
    def fn(batch):
        sent_lists, labels, lens_lists = zip(*batch)
        B = len(labels)

        x = torch.full((B, max_sents, max_tokens_per_sent), pad_id, dtype=torch.long)
        sent_lengths = torch.zeros((B, max_sents), dtype=torch.long)
        num_sents = torch.zeros((B,), dtype=torch.long)

        for i in range(B):
            sents = sent_lists[i]
            lens = lens_lists[i]
            S = min(len(sents), max_sents)
            num_sents[i] = S

            for j in range(S):
                ids = sents[j]
                L = min(ids.size(0), max_tokens_per_sent)
                x[i, j, :L] = ids[:L]
                sent_lengths[i, j] = L

        return x, torch.tensor(labels, dtype=torch.long), sent_lengths, num_sents
    return fn


def make_loaders_sentence(
    Xtr, ytr, Xv, yv, Xt, yt,
    stoi, unk_id, pad_id,
    max_sents, max_tokens_per_sent,
    batch
):
    train_loader = DataLoader(
        NewsSentenceDataset(Xtr, ytr, stoi, unk_id, max_sents, max_tokens_per_sent),
        batch_size=batch,
        shuffle=True,
        collate_fn=collate_sentence_batch(pad_id, max_sents, max_tokens_per_sent),
    )
    val_loader = DataLoader(
        NewsSentenceDataset(Xv, yv, stoi, unk_id, max_sents, max_tokens_per_sent),
        batch_size=batch,
        shuffle=False,
        collate_fn=collate_sentence_batch(pad_id, max_sents, max_tokens_per_sent),
    )
    test_loader = DataLoader(
        NewsSentenceDataset(Xt, yt, stoi, unk_id, max_sents, max_tokens_per_sent),
        batch_size=batch,
        shuffle=False,
        collate_fn=collate_sentence_batch(pad_id, max_sents, max_tokens_per_sent),
    )
    return train_loader, val_loader, test_loader
