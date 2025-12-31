import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

from model import BiLSTMWithGlove
from data_pipeline import *
import config



def evaluate(model, loader):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    preds_all, gold_all = [], []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, lengths in loader:
            x, y, lengths = x.to(config.device), y.to(config.device), lengths.to(config.device)
            logits = model(x, lengths)
            loss = criterion(logits, y)

            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            preds_all.append(preds.cpu().numpy())
            gold_all.append(y.cpu().numpy())
            # break

    y_true = np.concatenate(gold_all)
    y_pred = np.concatenate(preds_all)

    return loss_sum / total, correct / total, y_true, y_pred



torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

Xtr, Xv, Xt, ytr, yv, yt, target_names, num_classes = load_splits(
    config.REMOVE_PARTS, config.SEED, config.VAL_SPLIT
)
print(f"Sizes  -> train: {len(Xtr)}, val: {len(Xv)}, test: {len(Xt)}")
print(f"Labels -> train: {len(ytr)}, val: {len(yv)}, test: {len(yt)}")


stoi, itos, pad_id, unk_id, vocab_size = build_vocab(
    Xtr, config.MIN_FREQ, config.MAX_VOCAB
)

glove = load_glove(config.GLOVE_PATH, config.EMBED_DIM)
embedding_matrix, _ = build_embedding_matrix(
    stoi, pad_id, glove, config.EMBED_DIM
)

_, _, test_loader = make_loaders(
    Xtr, ytr, Xv, yv, Xt, yt,
    stoi, unk_id, pad_id, config.MAX_LEN, config.BATCH_SIZE
)

model = BiLSTMWithGlove(
    vocab_size, config.EMBED_DIM, config.HIDDEN_DIM,
    num_classes, pad_id, embedding_matrix,
    config.DROPOUT, config.FREEZE_EMBEDDINGS
).to(config.device)

model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.device))

test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader)

prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

print("\nTEST RESULTS")
print(f"Loss: {test_loss:.4f}")
print(f"Acc : {test_acc:.4f}")
print(f"Macro Precision: {prec:.4f}")
print(f"Macro Recall   : {rec:.4f}")
print(f"Macro F1       : {f1:.4f}")

with open(config.TEST_CSV, "w", newline="") as f:
    csv.writer(f).writerows([
        ["loss", "accuracy", "precision", "recall", "f1"],
        [test_loss, test_acc, prec, rec, f1]
    ])

cm = confusion_matrix(y_true, y_pred)
with open(config.CONFUSION_CSV, "w", newline="") as f:
    csv.writer(f).writerows(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))
