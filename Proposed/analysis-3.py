import os
import csv
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config
from model import BiLSTMWithGlove
from data_pipeline import (
    load_splits, build_vocab, load_glove, build_embedding_matrix,
    make_loaders_sentence
)

REPORT_DIR = "attention_reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def next_analysis_index():
    n = 1
    while True:
        p = os.path.join(REPORT_DIR, f"analysis-{n}.json")
        if not os.path.exists(p):
            return n
        n += 1


def entropy_from_probs(p, eps=1e-12):
    p = p.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1)


def sentence_importance_from_cross(doc_attn_weights, num_sents, K):
    ns = int(num_sents)
    if ns <= 0:
        return None
    qn = ns * int(K)
    if qn <= 0:
        return None
    attn = doc_attn_weights[:qn, :ns]        # (qn, ns)
    agg = attn.mean(dim=0)                   # (ns,)
    agg_sum = agg.sum().clamp(min=1e-12)
    imp = agg / agg_sum
    return imp


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def plot_histogram(values_a, values_b, title, out_path, label_a="correct", label_b="incorrect", bins=40):
    plt.figure()
    plt.hist(values_a, bins=bins, alpha=0.7, label=label_a)
    plt.hist(values_b, bins=bins, alpha=0.7, label=label_b)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@torch.no_grad()
def forward_with_topk(model, x, y, sent_lengths, num_sents, topk_value):
    old_topk = model.word_filter.topk
    model.word_filter.topk = int(topk_value)

    dbg = model(x, sent_lengths, num_sents, return_debug=True)
    logits = dbg["logits"]
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
    correct_mask = (pred == y)

    model.word_filter.topk = old_topk
    return pred, conf, correct_mask, dbg


def main():
    # Repro
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    # Load data
    Xtr, Xv, Xt, ytr, yv, yt, target_names, num_classes = load_splits(
        config.REMOVE_PARTS, config.SEED, config.VAL_SPLIT
    )

    # Vocab + embeddings
    stoi, itos, pad_id, unk_id, vocab_size = build_vocab(
        Xtr, config.MIN_FREQ, config.MAX_VOCAB
    )
    glove = load_glove(config.GLOVE_PATH, config.EMBED_DIM)
    embedding_matrix, coverage = build_embedding_matrix(
        stoi, pad_id, glove, config.EMBED_DIM
    )

    # Loaders
    _, _, test_loader = make_loaders_sentence(
        Xtr, ytr, Xv, yv, Xt, yt,
        stoi, unk_id, pad_id,
        config.MAX_SENTS, config.MAX_TOKENS_PER_SENT,
        config.BATCH_SIZE
    )

    TRAIN_TOPK = 5
    model = BiLSTMWithGlove(
        vocab_size, config.EMBED_DIM, config.HIDDEN_DIM,
        num_classes, pad_id, embedding_matrix,
        config.DROPOUT, config.FREEZE_EMBEDDINGS,
        topk_words=TRAIN_TOPK,
        attn_dropout=0.1
    ).to(config.device)


    state = torch.load(config.BEST_MODEL_PATH, map_location=config.device)
    model.load_state_dict(state)
    model.eval()


    TOPK_SWEEP = [1, 3, 5, 10, 20, 50]  # 50 ~= MAX_TOKENS_PER_SENT in your config
    TOPK_SWEEP = [k for k in TOPK_SWEEP if k <= config.MAX_TOKENS_PER_SENT]


    MAX_EXAMPLES_FOR_SWEEP = None
    MAX_EXAMPLES_FOR_STATS = None


    sweep_counts = {k: {"n": 0, "correct": 0, "agree_with_baseline": 0, "conf_sum": 0.0} for k in TOPK_SWEEP}
    baseline_total = 0
    baseline_correct = 0

    processed = 0
    for (x, y, sent_lengths, num_sents) in test_loader:
        x = x.to(config.device)
        y = y.to(config.device)
        sent_lengths = sent_lengths.to(config.device)
        num_sents = num_sents.to(config.device)

        # Baseline
        base_pred, base_conf, base_ok, _ = forward_with_topk(
            model, x, y, sent_lengths, num_sents, topk_value=TRAIN_TOPK
        )

        B = y.size(0)
        baseline_total += B
        baseline_correct += int(base_ok.sum().item())

        for k in TOPK_SWEEP:
            pred_k, conf_k, ok_k, _ = forward_with_topk(
                model, x, y, sent_lengths, num_sents, topk_value=k
            )
            sweep_counts[k]["n"] += B
            sweep_counts[k]["correct"] += int(ok_k.sum().item())
            sweep_counts[k]["agree_with_baseline"] += int((pred_k == base_pred).sum().item())
            sweep_counts[k]["conf_sum"] += float(conf_k.sum().item())

        processed += B
        if MAX_EXAMPLES_FOR_SWEEP is not None and processed >= MAX_EXAMPLES_FOR_SWEEP:
            break

    sweep_rows = []
    for k in TOPK_SWEEP:
        n = sweep_counts[k]["n"]
        acc = sweep_counts[k]["correct"] / max(1, n)
        agree = sweep_counts[k]["agree_with_baseline"] / max(1, n)
        avg_conf = sweep_counts[k]["conf_sum"] / max(1, n)
        sweep_rows.append([k, n, acc, agree, avg_conf])

    sweep_csv = os.path.join(REPORT_DIR, "analysis-3_topk_sensitivity.csv")
    save_csv(
        sweep_csv,
        header=["topk", "num_examples", "accuracy", "agreement_with_baseline", "avg_pred_confidence"],
        rows=sweep_rows
    )

    # Plot accuracy vs topk
    plt.figure()
    plt.plot([r[0] for r in sweep_rows], [r[2] for r in sweep_rows], marker="o")
    plt.title("Accuracy vs TOPK (word filtering strength)")
    plt.xlabel("TOPK")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    acc_plot = os.path.join(REPORT_DIR, "analysis-3_accuracy_vs_topk.png")
    plt.savefig(acc_plot)
    plt.close()

    baseline_acc = baseline_correct / max(1, baseline_total)


    # Attention statistics

    word_entropy_correct = []
    word_entropy_incorrect = []
    cross_entropy_correct = []
    cross_entropy_incorrect = []
    sent_max_correct = []
    sent_max_incorrect = []
    sent_entropy_correct = []
    sent_entropy_incorrect = []

    processed_stats = 0
    for (x, y, sent_lengths, num_sents) in test_loader:
        x = x.to(config.device)
        y = y.to(config.device)
        sent_lengths = sent_lengths.to(config.device)
        num_sents = num_sents.to(config.device)

        pred, conf, ok, dbg = forward_with_topk(
            model, x, y, sent_lengths, num_sents, topk_value=TRAIN_TOPK
        )

        # word-level attention entropy: attn_words is (B,S,T) softmax over tokens
        attn_words = dbg["attn_words"]  # (B,S,T)
        real_sent_mask = (sent_lengths > 0).float()  # (B,S)
        ent_word_per_sent = entropy_from_probs(attn_words)  # (B,S)
        # average entropy over real sentences for each doc
        denom = real_sent_mask.sum(dim=1).clamp(min=1.0)
        ent_word_doc = (ent_word_per_sent * real_sent_mask).sum(dim=1) / denom  # (B,)

        # cross-attention entropy
        doc_attn = dbg["doc_attn_weights"]  # (B, S*K, S)
        ent_cross_per_query = entropy_from_probs(doc_attn)  # (B, S*K)

        B, SK, Smax = doc_attn.shape
        K = int(dbg["topk_idx"].shape[-1])

        for b in range(B):
            ns = int(dbg["num_sents"][b].item())
            if ns <= 0:
                continue
            qn = ns * K
            # Cross entropy averaged over real queries
            ent_cross_doc = ent_cross_per_query[b, :qn].mean().item()

            # Sentence importance distribution
            imp = sentence_importance_from_cross(doc_attn[b], ns, K)
            if imp is None:
                continue
            imp = imp.detach()
            imp_ent = entropy_from_probs(imp.unsqueeze(0)).item()  # entropy of sentence distribution
            imp_max = float(imp.max().item())

            if bool(ok[b].item()):
                word_entropy_correct.append(float(ent_word_doc[b].item()))
                cross_entropy_correct.append(float(ent_cross_doc))
                sent_entropy_correct.append(float(imp_ent))
                sent_max_correct.append(float(imp_max))
            else:
                word_entropy_incorrect.append(float(ent_word_doc[b].item()))
                cross_entropy_incorrect.append(float(ent_cross_doc))
                sent_entropy_incorrect.append(float(imp_ent))
                sent_max_incorrect.append(float(imp_max))

        processed_stats += B
        if MAX_EXAMPLES_FOR_STATS is not None and processed_stats >= MAX_EXAMPLES_FOR_STATS:
            break

    # Save attention stats summary CSV
    def mean_std(xs):
        if len(xs) == 0:
            return (None, None)
        x = np.array(xs, dtype=np.float64)
        return float(x.mean()), float(x.std(ddof=0))

    stats_rows = []
    for name, a, b in [
        ("word_entropy", word_entropy_correct, word_entropy_incorrect),
        ("cross_entropy", cross_entropy_correct, cross_entropy_incorrect),
        ("sentence_importance_entropy", sent_entropy_correct, sent_entropy_incorrect),
        ("sentence_importance_max", sent_max_correct, sent_max_incorrect),
    ]:
        m1, s1 = mean_std(a)
        m2, s2 = mean_std(b)
        stats_rows.append([name, len(a), m1, s1, len(b), m2, s2])

    stats_csv = os.path.join(REPORT_DIR, "analysis-3_attention_stats.csv")
    save_csv(
        stats_csv,
        header=["metric", "n_correct", "mean_correct", "std_correct", "n_incorrect", "mean_incorrect", "std_incorrect"],
        rows=stats_rows
    )

    # Plots: distributions (correct vs incorrect)
    plot_histogram(
        word_entropy_correct, word_entropy_incorrect,
        title="Word-attention entropy (per-doc avg over sentences): correct vs incorrect",
        out_path=os.path.join(REPORT_DIR, "analysis-3_entropy_word_correct_incorrect.png")
    )
    plot_histogram(
        cross_entropy_correct, cross_entropy_incorrect,
        title="Cross-attention entropy (per-doc avg over queries): correct vs incorrect",
        out_path=os.path.join(REPORT_DIR, "analysis-3_entropy_cross_correct_incorrect.png")
    )
    plot_histogram(
        sent_max_correct, sent_max_incorrect,
        title="Max sentence-importance mass: correct vs incorrect",
        out_path=os.path.join(REPORT_DIR, "analysis-3_sentence_importance_max_correct_incorrect.png")
    )


    analysis_idx = next_analysis_index()
    out_json = os.path.join(REPORT_DIR, f"analysis-{analysis_idx}.json")

    payload = {
        "meta": {
            "analysis_id": f"analysis-{analysis_idx}",
            "created_at": datetime.now().isoformat(),
            "split": "test",
            "model_checkpoint": config.BEST_MODEL_PATH,
            "glove_coverage": float(coverage),
            "train_topk": int(TRAIN_TOPK),
            "topk_sweep": TOPK_SWEEP,
            "limits": {
                "MAX_EXAMPLES_FOR_SWEEP": MAX_EXAMPLES_FOR_SWEEP,
                "MAX_EXAMPLES_FOR_STATS": MAX_EXAMPLES_FOR_STATS
            },
            "files": {
                "topk_sensitivity_csv": os.path.basename(sweep_csv),
                "attention_stats_csv": os.path.basename(stats_csv),
                "accuracy_vs_topk_png": "analysis-3_accuracy_vs_topk.png",
                "word_entropy_hist_png": "analysis-3_entropy_word_correct_incorrect.png",
                "cross_entropy_hist_png": "analysis-3_entropy_cross_correct_incorrect.png",
                "sentence_max_hist_png": "analysis-3_sentence_importance_max_correct_incorrect.png",
            }
        },
        "baseline": {
            "baseline_topk": int(TRAIN_TOPK),
            "accuracy": float(baseline_acc),
            "num_examples": int(baseline_total)
        },
        "topk_sensitivity": [
            {
                "topk": int(r[0]),
                "num_examples": int(r[1]),
                "accuracy": float(r[2]),
                "agreement_with_baseline": float(r[3]),
                "avg_pred_confidence": float(r[4]),
            }
            for r in sweep_rows
        ],
        "attention_statistics_summary": [
            {
                "metric": row[0],
                "n_correct": int(row[1]),
                "mean_correct": row[2],
                "std_correct": row[3],
                "n_incorrect": int(row[4]),
                "mean_incorrect": row[5],
                "std_incorrect": row[6],
            }
            for row in stats_rows
        ],
        "interpretation_notes": {
            "how_to_use_in_report": [
                "If accuracy drops when TOPK is very small (e.g., 1), it suggests aggressive filtering removes useful evidence.",
                "If accuracy drops when TOPK is very large (e.g., 50), it suggests filtering helps reduce noise / distractor words.",
                "Lower attention entropy often indicates more peaked attention; compare correct vs incorrect to see whether over-peaked attention correlates with failures.",
                "Higher max sentence-importance mass can indicate the model over-relies on a single sentence; if this is higher for incorrect cases, it supports a negative impact claim."
            ]
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
