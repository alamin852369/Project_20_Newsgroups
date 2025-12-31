# train.py
import csv
import numpy as np
import torch
import torch.nn as nn

from model import BiLSTMWithGlove
from data_pipeline import *
import config




def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y, lengths in loader:
            x, y, lengths = x.to(config.device), y.to(config.device), lengths.to(config.device)
            logits = model(x, lengths)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total



torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

Xtr, Xv, _, ytr, yv, _, _, num_classes = load_splits(
    config.REMOVE_PARTS, config.SEED, config.VAL_SPLIT
)


print(f"Sizes -> train: {len(Xtr)}, val: {len(Xv)}")
print(f"Labels -> train: {len(ytr)}, val: {len(yv)}")



stoi, itos, pad_id, unk_id, vocab_size = build_vocab(
    Xtr, config.MIN_FREQ, config.MAX_VOCAB
)

glove = load_glove(config.GLOVE_PATH, config.EMBED_DIM)
embedding_matrix, coverage = build_embedding_matrix(
    stoi, pad_id, glove, config.EMBED_DIM
)

print(f"GloVe coverage: {coverage:.2%}")

train_loader, val_loader, _ = make_loaders(
    Xtr, ytr, Xv, yv, [], [],
    stoi, unk_id, pad_id, config.MAX_LEN, config.BATCH_SIZE
)

model = BiLSTMWithGlove(
    vocab_size, config.EMBED_DIM, config.HIDDEN_DIM,
    num_classes, pad_id, embedding_matrix,
    config.DROPOUT, config.FREEZE_EMBEDDINGS
).to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
)

best_val_loss = float("inf")
history = []

for epoch in range(1, config.EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, train=True)
    va_loss, va_acc = run_epoch(model, val_loader, optimizer, criterion, train=False)

    history.append([epoch, tr_loss, tr_acc, va_loss, va_acc])
    print(f"Epoch {epoch:03d} | Train {tr_loss:.4f}/{tr_acc:.4f} | Val {va_loss:.4f}/{va_acc:.4f}")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        torch.save(model.state_dict(), config.BEST_MODEL_PATH)

# Save training history
with open(config.HISTORY_CSV, "w", newline="") as f:
    csv.writer(f).writerows(
        [["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]] + history
    )

print("Training finished. Best model saved.")
