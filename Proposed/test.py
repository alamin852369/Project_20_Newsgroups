import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

from model import BiLSTMWithGlove
from data_pipeline import (
    load_splits, build_vocab, load_glove, build_embedding_matrix, make_loaders_sentence
)
import config


def save_confusion_matrix_plot(cm, class_names, out_path, title, normalize=False):
    cm_to_plot = cm.astype(np.float64)

    if normalize:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_to_plot = cm_to_plot / row_sums

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_to_plot, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    # ticks
    n = len(class_names)
    plt.xticks(range(n), class_names, rotation=90)
    plt.yticks(range(n), class_names)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_macro_metrics_bar(prec, rec, f1, out_path):
    labels = ["Macro Precision", "Macro Recall", "Macro F1"]
    values = [prec, rec, f1]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.title("Macro metrics on test set")
    plt.ylabel("score")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_per_class_f1_bar(y_true, y_pred, class_names, out_path):
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    plt.figure(figsize=(12, 4))
    plt.bar(class_names, f1)
    plt.ylim(0.0, 1.0)
    plt.title("Per-class F1 on test set")
    plt.ylabel("F1")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds_all, gold_all = [], []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, sent_lengths, num_sents in loader:
            x = x.to(device)
            y = y.to(device)
            sent_lengths = sent_lengths.to(device)
            num_sents = num_sents.to(device)

            logits = model(x, sent_lengths, num_sents)
            loss = criterion(logits, y)

            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            preds_all.append(preds.cpu().numpy())
            gold_all.append(y.cpu().numpy())

    y_true = np.concatenate(gold_all)
    y_pred = np.concatenate(preds_all)

    return loss_sum / total, correct / total, y_true, y_pred


torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

Xtr, Xv, Xt, ytr, yv, yt, target_names, num_classes = load_splits(
    config.REMOVE_PARTS, config.SEED, config.VAL_SPLIT
)

stoi, itos, pad_id, unk_id, vocab_size = build_vocab(
    Xtr, config.MIN_FREQ, config.MAX_VOCAB
)

glove = load_glove(config.GLOVE_PATH, config.EMBED_DIM)
embedding_matrix, _ = build_embedding_matrix(
    stoi, pad_id, glove, config.EMBED_DIM
)

MAX_SENTS = getattr(config, "MAX_SENTS", 20)
MAX_TOKENS_PER_SENT = getattr(config, "MAX_TOKENS_PER_SENT", 50)

train_loader, val_loader, test_loader = make_loaders_sentence(
    Xtr, ytr, Xv, yv, Xt, yt,
    stoi, unk_id, pad_id,
    MAX_SENTS, MAX_TOKENS_PER_SENT,
    config.BATCH_SIZE
)


TOPK_WORDS = getattr(config, "TOPK_WORDS", 5)

model = BiLSTMWithGlove(
    vocab_size=vocab_size,
    embed_dim=config.EMBED_DIM,
    hidden_dim=config.HIDDEN_DIM,
    num_classes=num_classes,
    pad_id=pad_id,
    embedding_matrix=embedding_matrix,
    dropout=config.DROPOUT,
    freeze=config.FREEZE_EMBEDDINGS,
    topk_words=TOPK_WORDS,
    attn_dropout=0.1
).to(config.device)

state = torch.load(config.BEST_MODEL_PATH, map_location=config.device)
model.load_state_dict(state)

test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, config.device)

prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

# Save numeric metrics CSV
with open(config.TEST_CSV, "w", newline="") as f:
    csv.writer(f).writerows([
        ["loss", "accuracy", "macro_precision", "macro_recall", "macro_f1"],
        [test_loss, test_acc, prec, rec, f1]
    ])

# Confusion CSV
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
with open(config.CONFUSION_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["true/pred"] + target_names)
    for i in range(num_classes):
        w.writerow([target_names[i]] + cm[i].tolist())

# Classification report text file
report_txt = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
with open("test_classification_report.txt", "w", encoding="utf-8") as f:
    f.write("TEST RESULTS\n")
    f.write(f"Loss: {test_loss:.4f}\n")
    f.write(f"Acc : {test_acc:.4f}\n")
    f.write(f"Macro Precision: {prec:.4f}\n")
    f.write(f"Macro Recall   : {rec:.4f}\n")
    f.write(f"Macro F1       : {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_txt)


save_confusion_matrix_plot(
    cm=cm,
    class_names=target_names,
    out_path="test_confusion_matrix_counts.png",
    title="Confusion Matrix (Counts)",
    normalize=False
)

save_confusion_matrix_plot(
    cm=cm,
    class_names=target_names,
    out_path="test_confusion_matrix_normalized.png",
    title="Confusion Matrix (Row-normalized)",
    normalize=True
)

save_macro_metrics_bar(
    prec=prec, rec=rec, f1=f1,
    out_path="test_macro_metrics_bar.png"
)

save_per_class_f1_bar(
    y_true=y_true, y_pred=y_pred,
    class_names=target_names,
    out_path="test_per_class_f1_bar.png"
)
