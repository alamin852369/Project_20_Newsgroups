import os
import json
import csv
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib.pyplot as plt

import config
from model import BiLSTMWithGlove
from data_pipeline import (
    load_splits, build_vocab, load_glove, build_embedding_matrix,
    make_loaders_sentence
)

REPORT_DIR = "attention_reports"
ERR_DIR = os.path.join(REPORT_DIR, "analysis-2_errors")
PLOTS_DIR = os.path.join(REPORT_DIR, "analysis-2_plots")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ERR_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def decode_tokens(token_ids_1d, itos, pad_id):
    toks = []
    for tid in token_ids_1d.tolist():
        if tid == pad_id:
            break
        if 0 <= tid < len(itos):
            toks.append(itos[tid])
        else:
            toks.append("<UNK>")
    return toks


def safe_float(x):
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, np.generic):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def next_available_path(base_dir, base_name, ext):
    p1 = os.path.join(base_dir, f"{base_name}.{ext}")
    if not os.path.exists(p1):
        return p1
    i = 2
    while True:
        pi = os.path.join(base_dir, f"{base_name}-{i}.{ext}")
        if not os.path.exists(pi):
            return pi
        i += 1


def save_confusion_heatmap(cm, class_names, out_path, title):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90, fontsize=7)
    plt.yticks(ticks, class_names, fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_bar(values, labels, out_path, title, xlabel, ylabel, rotate=90, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=rotate, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def compute_sentence_importance(doc_attn_weights, num_sents, K):
    ns = num_sents
    qn = ns * K
    if qn <= 0 or ns <= 0:
        return []

    attn = doc_attn_weights[:qn, :ns]
    agg = attn.mean(dim=0)
    agg_sum = agg.sum().clamp(min=1e-12)
    agg_norm = (agg / agg_sum).detach().cpu().tolist()

    ranked = sorted(
        [{"sentence_id": i, "importance": float(agg_norm[i])} for i in range(ns)],
        key=lambda d: d["importance"],
        reverse=True
    )
    return ranked


@torch.no_grad()
def extract_error_example(model, x, y, sent_lengths, num_sents, itos, pad_id, target_names,
                          top_sentences_to_keep=5,
                          top_words_per_sentence=5,
                          top_sentence_links_per_query=2):

    dbg = model(x, sent_lengths, num_sents, return_debug=True)

    logits = dbg["logits"][0]
    probs = F.softmax(logits, dim=-1)
    pred_id = int(probs.argmax().item())
    conf = float(probs[pred_id].item())
    true_id = int(y[0].item())

    scores = dbg["scores"][0]
    attn_words = dbg["attn_words"][0]
    topk_idx = dbg["topk_idx"][0]
    doc_attn = dbg["doc_attn_weights"][0]
    K = topk_idx.shape[-1]
    ns = int(dbg["num_sents"][0].item())

    S = x.shape[1]
    decoded = []
    for s in range(S):
        decoded.append(decode_tokens(x[0, s], itos, pad_id))

    ranked = compute_sentence_importance(doc_attn, ns, K)
    top_ranked = ranked[:top_sentences_to_keep]

    sentence_details = []
    for item in top_ranked:
        s = item["sentence_id"]
        sent_tokens = decoded[s]
        pos_list = topk_idx[s].tolist()

        top_words = []
        for r, pos in enumerate(pos_list[:top_words_per_sentence]):
            tok = sent_tokens[pos] if pos < len(sent_tokens) else "<PAD/OUT>"
            top_words.append({
                "rank": int(r),
                "token": tok,
                "token_pos": int(pos),
                "word_score_logit": safe_float(scores[s, pos]),
                "word_attn_weight": safe_float(attn_words[s, pos]),
            })

        sentence_details.append({
            "sentence_id": int(s),
            "importance": float(item["importance"]),
            "sentence_text": " ".join(sent_tokens),
            "top_words": top_words
        })

    query_links = []
    if ns > 0:
        important_sentence_ids = [d["sentence_id"] for d in top_ranked[:2]]
        for s in important_sentence_ids:
            for k_i in range(min(K, top_words_per_sentence)):
                q = s * K + k_i
                pos = int(topk_idx[s, k_i].item())
                tok = decoded[s][pos] if pos < len(decoded[s]) else "<PAD/OUT>"

                attn_row = doc_attn[q, :ns]
                vals, idx = torch.topk(attn_row, k=min(top_sentence_links_per_query, ns))
                links = []
                for rr in range(idx.numel()):
                    sid = int(idx[rr].item())
                    links.append({
                        "rank": int(rr),
                        "sentence_id": sid,
                        "attn_weight": safe_float(vals[rr]),
                        "sentence_text": " ".join(decoded[sid])
                    })

                query_links.append({
                    "query_id": int(q),
                    "from_sentence_id": int(s),
                    "token": tok,
                    "token_pos": int(pos),
                    "top_attended_sentences": links
                })

    return {
        "true_label_id": true_id,
        "pred_label_id": pred_id,
        "true_label_name": target_names[true_id] if target_names else str(true_id),
        "pred_label_name": target_names[pred_id] if target_names else str(pred_id),
        "pred_confidence": conf,
        "num_sents": ns,
        "top_sentences": sentence_details,
        "sample_query_to_sentence_links": query_links,
    }


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    Xtr, Xv, Xt, ytr, yv, yt, target_names, num_classes = load_splits(
        config.REMOVE_PARTS, config.SEED, config.VAL_SPLIT
    )

    stoi, itos, pad_id, unk_id, vocab_size = build_vocab(
        Xtr, config.MIN_FREQ, config.MAX_VOCAB
    )
    glove = load_glove(config.GLOVE_PATH, config.EMBED_DIM)
    embedding_matrix, coverage = build_embedding_matrix(
        stoi, pad_id, glove, config.EMBED_DIM
    )

    _, _, test_loader = make_loaders_sentence(
        Xtr, ytr, Xv, yv, Xt, yt,
        stoi, unk_id, pad_id,
        config.MAX_SENTS, config.MAX_TOKENS_PER_SENT,
        config.BATCH_SIZE
    )

    TOPK_WORDS = 5
    model = BiLSTMWithGlove(
        vocab_size, config.EMBED_DIM, config.HIDDEN_DIM,
        num_classes, pad_id, embedding_matrix,
        config.DROPOUT, config.FREEZE_EMBEDDINGS,
        topk_words=TOPK_WORDS,
        attn_dropout=0.1
    ).to(config.device)

    state = torch.load(config.BEST_MODEL_PATH, map_location=config.device)
    model.load_state_dict(state)
    model.eval()

    all_true = []
    all_pred = []
    all_conf = []

    misclassified_store = []
    ex_counter = 0

    with torch.no_grad():
        for _, (x, y, sent_lengths, num_sents) in enumerate(test_loader):
            x = x.to(config.device)
            y = y.to(config.device)
            sent_lengths = sent_lengths.to(config.device)
            num_sents = num_sents.to(config.device)

            logits = model(x, sent_lengths, num_sents)     # (B,C)
            probs = F.softmax(logits, dim=-1)              # (B,C)
            pred = probs.argmax(dim=-1)                    # (B,)
            conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)

            bt = y.detach().cpu().tolist()
            bp = pred.detach().cpu().tolist()
            bc = conf.detach().cpu().tolist()

            all_true.extend(bt)
            all_pred.extend(bp)
            all_conf.extend(bc)

            B = y.size(0)
            for j in range(B):
                if bp[j] != bt[j]:
                    example_id = f"test_{ex_counter}"
                    misclassified_store.append({
                        "example_id": example_id,
                        "true": int(bt[j]),
                        "pred": int(bp[j]),
                        "conf": float(bc[j]),
                        "x": x[j].detach().cpu(),
                        "y": y[j].detach().cpu(),
                        "sent_lengths": sent_lengths[j].detach().cpu(),
                        "num_sents": num_sents[j].detach().cpu(),
                    })
                ex_counter += 1

    acc = float(accuracy_score(all_true, all_pred))
    pr, rc, f1, sup = precision_recall_fscore_support(
        all_true, all_pred, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))

    per_class = []
    for c in range(num_classes):
        per_class.append({
            "class_id": c,
            "class_name": target_names[c] if target_names else str(c),
            "precision": float(pr[c]),
            "recall": float(rc[c]),
            "f1": float(f1[c]),
            "support": int(sup[c]),
        })

    hardest = sorted(per_class, key=lambda d: (d["f1"], d["support"]))

    conf_pairs = []
    for t in range(num_classes):
        for p in range(num_classes):
            if t == p:
                continue
            if cm[t, p] > 0:
                conf_pairs.append({
                    "true_id": t,
                    "pred_id": p,
                    "true_name": target_names[t] if target_names else str(t),
                    "pred_name": target_names[p] if target_names else str(p),
                    "count": int(cm[t, p])
                })
    conf_pairs.sort(key=lambda d: d["count"], reverse=True)

    TOP_PAIRS = 10
    EXAMPLES_PER_PAIR = 5

    by_pair = defaultdict(list)
    for m in misclassified_store:
        by_pair[(m["true"], m["pred"])].append(m)

    selected = []
    for pair in conf_pairs[:TOP_PAIRS]:
        t, p = pair["true_id"], pair["pred_id"]
        candidates = sorted(by_pair.get((t, p), []), key=lambda d: d["conf"], reverse=True)
        selected.extend(candidates[:EXAMPLES_PER_PAIR])

    LOW_CONF_EXTRA = 10
    low_conf = sorted(misclassified_store, key=lambda d: d["conf"])[:LOW_CONF_EXTRA]
    sel_ids = set([d["example_id"] for d in selected])
    for d in low_conf:
        if d["example_id"] not in sel_ids:
            selected.append(d)
            sel_ids.add(d["example_id"])

    detailed_errors_index = []
    for item in selected:
        x1 = item["x"].unsqueeze(0).to(config.device)
        y1 = item["y"].unsqueeze(0).to(config.device)
        sl1 = item["sent_lengths"].unsqueeze(0).to(config.device)
        ns1 = item["num_sents"].unsqueeze(0).to(config.device)

        detail = extract_error_example(
            model=model,
            x=x1, y=y1,
            sent_lengths=sl1, num_sents=ns1,
            itos=itos, pad_id=pad_id,
            target_names=target_names,
            top_sentences_to_keep=5,
            top_words_per_sentence=TOPK_WORDS,
            top_sentence_links_per_query=2
        )
        detail["example_id"] = item["example_id"]

        per_path = os.path.join(ERR_DIR, f"{item['example_id']}.json")
        with open(per_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, ensure_ascii=False)

        detailed_errors_index.append({
            "example_id": item["example_id"],
            "true_label_name": detail["true_label_name"],
            "pred_label_name": detail["pred_label_name"],
            "pred_confidence": detail["pred_confidence"],
            "num_sents": detail["num_sents"],
            "top_sentences_preview": [
                {"sentence_id": s["sentence_id"], "importance": s["importance"]}
                for s in detail["top_sentences"][:3]
            ],
            "file": f"analysis-2_errors/{item['example_id']}.json"
        })

    cm_csv_path = next_available_path(REPORT_DIR, "analysis-2_confusion", "csv")
    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["true/pred"] + [target_names[i] for i in range(num_classes)]
        w.writerow(header)
        for t in range(num_classes):
            row = [target_names[t]] + [int(cm[t, p]) for p in range(num_classes)]
            w.writerow(row)

    metrics_csv_path = next_available_path(REPORT_DIR, "analysis-2_per_class_metrics", "csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
        for d in per_class:
            w.writerow([d["class_id"], d["class_name"], d["precision"], d["recall"], d["f1"], d["support"]])

    out_json_path = next_available_path(REPORT_DIR, "analysis-2", "json")

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "split": "test",
            "model_checkpoint": config.BEST_MODEL_PATH,
            "glove_coverage": float(coverage),
            "settings": {
                "MAX_SENTS": int(config.MAX_SENTS),
                "MAX_TOKENS_PER_SENT": int(config.MAX_TOKENS_PER_SENT),
                "TOPK_WORDS": int(TOPK_WORDS),
                "TOP_PAIRS": int(TOP_PAIRS),
                "EXAMPLES_PER_PAIR": int(EXAMPLES_PER_PAIR),
                "LOW_CONF_EXTRA": int(LOW_CONF_EXTRA),
            }
        },
        "overall": {
            "accuracy": acc,
            "num_test_examples": int(len(all_true)),
            "num_misclassified": int(len(misclassified_store)),
        },
        "per_class_metrics": per_class,
        "hardest_topics_by_f1": hardest[:10],
        "most_confused_pairs": conf_pairs[:20],
        "selected_error_examples_index": detailed_errors_index,
        "saved_files": {
            "analysis_json": os.path.basename(out_json_path),
            "confusion_csv": os.path.basename(cm_csv_path),
            "per_class_metrics_csv": os.path.basename(metrics_csv_path),
            "errors_dir": os.path.basename(ERR_DIR),
            "plots_dir": os.path.basename(PLOTS_DIR),
        }
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    cm_png = next_available_path(PLOTS_DIR, "analysis-2_confusion_heatmap", "png")
    save_confusion_heatmap(cm, target_names, cm_png, "Confusion Matrix (Test)")

    f1_vals = [d["f1"] for d in per_class]
    f1_labels = [d["class_name"] for d in per_class]
    f1_png = next_available_path(PLOTS_DIR, "analysis-2_per_class_f1", "png")
    save_bar(
        f1_vals, f1_labels, f1_png,
        "Per-class F1 Score (Test)",
        "Class", "F1", rotate=90, figsize=(14, 6)
    )

    hard10 = hardest[:10]
    hard_vals = [d["f1"] for d in hard10]
    hard_labels = [d["class_name"] for d in hard10]
    hard_png = next_available_path(PLOTS_DIR, "analysis-2_hardest_10_by_f1", "png")
    save_bar(
        hard_vals, hard_labels, hard_png,
        "Hardest 10 Classes by F1 (Lower is harder)",
        "Class", "F1", rotate=45, figsize=(10, 6)
    )

    top_pairs = conf_pairs[:10]
    pair_labels = [f"{d['true_name']} -> {d['pred_name']}" for d in top_pairs]
    pair_vals = [d["count"] for d in top_pairs]
    pair_png = next_available_path(PLOTS_DIR, "analysis-2_top_confused_pairs", "png")
    save_bar(
        pair_vals, pair_labels, pair_png,
        "Top Confused True->Pred Pairs (Counts)",
        "Pair", "Count", rotate=45, figsize=(12, 6)
    )


if __name__ == "__main__":
    main()
