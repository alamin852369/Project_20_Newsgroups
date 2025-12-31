import os
import re
import json
import random
from datetime import datetime
from collections import Counter

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
CASE_DIR = os.path.join(REPORT_DIR, "analysis-4_cases")
PLOTS_DIR = os.path.join(REPORT_DIR, "analysis-4_plots")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CASE_DIR, exist_ok=True)
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


_WORD_RE = re.compile(r"^[A-Za-z]+$")
def is_word(tok: str) -> bool:
    return bool(_WORD_RE.match(tok))


def is_number(tok: str) -> bool:
    return tok.isdigit()


STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","to","of","in","on","for","with","as","by",
    "is","are","was","were","be","been","being","do","does","did","done",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","their","our",
    "this","that","these","those","there","here","from","at","into","out","up","down","over","under",
    "not","no","yes","can","could","should","would","will","just","so","than","too","very",
    "about","after","before","because","while","when","where","what","which","who","whom","why","how"
}


def token_quality(tok: str) -> str:
    t = tok.lower()
    if t == "<unk>":
        return "unk"
    if t in STOPWORDS:
        return "stopword"
    if is_number(t):
        return "number"
    if is_word(t) and len(t) <= 2:
        return "short_word"
    if is_word(t):
        return "content_word"
    return "other"



# Failure mode scoring
@torch.no_grad()
def analyze_one_example(dbg, x_b, sent_lengths_b, itos, pad_id):
    scores = dbg["scores"]
    attn_words = dbg["attn_words"]
    topk_idx = dbg["topk_idx"]
    doc_attn = dbg["doc_attn_weights"]
    num_sents = int(dbg["num_sents"].item())
    S, T = x_b.shape
    K = int(topk_idx.shape[-1])

    decoded = []
    for s in range(S):
        decoded.append(decode_tokens(x_b[s], itos, pad_id))

    ns = num_sents

    selected_qualities = []
    selected_by_sentence = []

    for s in range(ns):
        pos_list = topk_idx[s].tolist()
        toks = decoded[s]
        sent_sel = []
        for k_i in range(K):
            pos = pos_list[k_i]
            tok = toks[pos] if pos < len(toks) else "<PAD/OUT>"
            q = token_quality(tok)
            selected_qualities.append(q)
            sent_sel.append((tok, pos, q, safe_float(scores[s, pos]), safe_float(attn_words[s, pos])))
        selected_by_sentence.append(sent_sel)

    qual_counts = Counter(selected_qualities)
    total_sel = max(1, len(selected_qualities))
    stop_ratio = qual_counts["stopword"] / total_sel
    unk_ratio = qual_counts["unk"] / total_sel
    num_ratio = qual_counts["number"] / total_sel
    content_ratio = qual_counts["content_word"] / total_sel

    mismatch_count = 0
    total_sents_checked = 0
    for s in range(ns):
        toks = decoded[s]
        if len(toks) == 0:
            continue
        total_sents_checked += 1
        attn_best_pos = int(attn_words[s].argmax().item())
        if attn_best_pos not in set(topk_idx[s].tolist()):
            mismatch_count += 1
    attn_top_not_selected_rate = mismatch_count / max(1, total_sents_checked)

    qn = ns * K
    if qn > 0 and ns > 0:
        cross_ent = float(entropy_from_probs(doc_attn[:qn, :ns]).mean().item())
        sent_imp = doc_attn[:qn, :ns].mean(dim=0)
        sent_imp = sent_imp / sent_imp.sum().clamp(min=1e-12)
        sent_imp_ent = float(entropy_from_probs(sent_imp).item())
        sent_imp_max = float(sent_imp.max().item())
        top_sentence = int(sent_imp.argmax().item())
        sent_imp_list = sent_imp.detach().cpu().tolist()
    else:
        cross_ent, sent_imp_ent, sent_imp_max, top_sentence = 0.0, 0.0, 1.0, 0
        sent_imp_list = [1.0]

    mismatch_q = 0
    for s in range(ns):
        for k_i in range(K):
            q = s * K + k_i
            if q >= qn:
                continue
            best_sid = int(doc_attn[q, :ns].argmax().item())
            if best_sid != s:
                mismatch_q += 1
    query_mismatch_rate = mismatch_q / max(1, qn)

    metrics = {
        "num_sents": ns,
        "K": K,
        "stop_ratio": float(stop_ratio),
        "unk_ratio": float(unk_ratio),
        "num_ratio": float(num_ratio),
        "content_ratio": float(content_ratio),
        "attn_top_not_selected_rate": float(attn_top_not_selected_rate),
        "cross_entropy": float(cross_ent),
        "sentence_importance_entropy": float(sent_imp_ent),
        "sentence_importance_max": float(sent_imp_max),
        "top_sentence_id": int(top_sentence),
        "query_mismatch_rate": float(query_mismatch_rate),
        "selected_quality_counts": dict(qual_counts),
    }

    payload_bits = {
        "decoded_sentences": [" ".join(decoded[s]) for s in range(ns)],
        "selected_by_sentence": [
            [
                {
                    "token": tok,
                    "token_pos": int(pos),
                    "quality": q,
                    "word_score_logit": float(ws),
                    "word_attn_weight": float(wa),
                }
                for (tok, pos, q, ws, wa) in sent_sel
            ]
            for sent_sel in selected_by_sentence
        ],
        "top_sentence_id": int(top_sentence),
        "sentence_importance_distribution": sent_imp_list
    }

    return metrics, payload_bits


def classify_failure_modes(metrics, pred_conf, wrong=True):
    modes = {}

    modes["A_context_poor_topk"] = (
        0.6 * metrics["stop_ratio"] +
        0.3 * metrics["unk_ratio"] +
        0.1 * metrics["num_ratio"]
    )

    modes["B_word_selection_mismatch"] = metrics["attn_top_not_selected_rate"]

    modes["C_cross_attention_collapse"] = (
        0.7 * metrics["sentence_importance_max"] +
        0.3 * max(0.0, 1.5 - metrics["sentence_importance_entropy"]) / 1.5
    )

    modes["D_local_global_conflict"] = metrics["query_mismatch_rate"]

    modes["E_overconfident_error"] = float(pred_conf) if wrong else 0.0
    return modes



# Plot saving
def save_hist(values, title, xlabel, out_path, bins=40):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_bar(labels, values, title, ylabel, out_path):
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_scatter(x, y, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_corr_heatmap(mat, labels, title, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



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

    TOP_CASES_PER_MODE = 25
    MAX_SCAN_EXAMPLES = None

    candidates = []
    ex_id = 0

    with torch.no_grad():
        for (x, y, sent_lengths, num_sents) in test_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            sent_lengths = sent_lengths.to(config.device)
            num_sents = num_sents.to(config.device)

            dbg_batch = model(x, sent_lengths, num_sents, return_debug=True)
            logits = dbg_batch["logits"]
            probs = F.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)

            B = y.size(0)
            for b in range(B):
                eid = f"test_{ex_id}"
                ex_id += 1

                if MAX_SCAN_EXAMPLES is not None and ex_id > MAX_SCAN_EXAMPLES:
                    break

                true_id = int(y[b].item())
                pred_id = int(pred[b].item())
                pred_conf = float(conf[b].item())

                if pred_id == true_id:
                    continue

                dbg = {
                    "scores": dbg_batch["scores"][b],
                    "attn_words": dbg_batch["attn_words"][b],
                    "topk_idx": dbg_batch["topk_idx"][b],
                    "doc_attn_weights": dbg_batch["doc_attn_weights"][b],
                    "num_sents": dbg_batch["num_sents"][b],
                }

                metrics, bits = analyze_one_example(
                    dbg=dbg,
                    x_b=x[b].detach(),
                    sent_lengths_b=sent_lengths[b].detach(),
                    itos=itos,
                    pad_id=pad_id
                )

                modes = classify_failure_modes(metrics, pred_conf=pred_conf, wrong=True)

                candidates.append({
                    "example_id": eid,
                    "true_label_id": true_id,
                    "pred_label_id": pred_id,
                    "true_label_name": target_names[true_id],
                    "pred_label_name": target_names[pred_id],
                    "pred_confidence": pred_conf,
                    "metrics": metrics,
                    "mode_scores": modes,
                    "payload_bits": bits
                })

            if MAX_SCAN_EXAMPLES is not None and ex_id > MAX_SCAN_EXAMPLES:
                break

    mode_names = [
        "A_context_poor_topk",
        "B_word_selection_mismatch",
        "C_cross_attention_collapse",
        "D_local_global_conflict",
        "E_overconfident_error"
    ]

    # Save plots over all misclassified examples
    mode_matrix = []
    for c in candidates:
        mode_matrix.append([c["mode_scores"].get(m, 0.0) for m in mode_names])
    mode_matrix = np.array(mode_matrix, dtype=np.float64) if len(mode_matrix) else np.zeros((1, len(mode_names)))

    means = mode_matrix.mean(axis=0).tolist()
    save_bar(
        labels=mode_names,
        values=means,
        title="Mean failure-mode score over misclassified examples",
        ylabel="mean score",
        out_path=os.path.join(PLOTS_DIR, "analysis-4_mode_means.png")
    )

    for i, m in enumerate(mode_names):
        save_hist(
            values=mode_matrix[:, i].tolist(),
            title=f"Histogram of {m} scores over misclassified examples",
            xlabel="mode score",
            out_path=os.path.join(PLOTS_DIR, f"analysis-4_hist_{m}.png")
        )

    confs = [c["pred_confidence"] for c in candidates]
    for i, m in enumerate(mode_names):
        save_scatter(
            x=confs,
            y=mode_matrix[:, i].tolist(),
            title=f"Confidence vs {m}",
            xlabel="pred_confidence",
            ylabel=m,
            out_path=os.path.join(PLOTS_DIR, f"analysis-4_scatter_conf_vs_{m}.png")
        )

    corr = np.corrcoef(mode_matrix.T) if mode_matrix.shape[0] > 2 else np.eye(len(mode_names))
    save_corr_heatmap(
        mat=corr,
        labels=mode_names,
        title="Correlation between failure-mode scores",
        out_path=os.path.join(PLOTS_DIR, "analysis-4_mode_correlation.png")
    )

    # Select top cases per mode
    selected_by_mode = {}
    for mode in mode_names:
        ranked = sorted(candidates, key=lambda d: d["mode_scores"].get(mode, 0.0), reverse=True)
        selected_by_mode[mode] = ranked[:TOP_CASES_PER_MODE]

    saved_ids = set()
    case_index = []
    for mode, items in selected_by_mode.items():
        for item in items:
            if item["example_id"] in saved_ids:
                continue
            saved_ids.add(item["example_id"])

            path = os.path.join(CASE_DIR, f"{item['example_id']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(item, f, indent=2, ensure_ascii=False)

            case_index.append({
                "example_id": item["example_id"],
                "true_label_name": item["true_label_name"],
                "pred_label_name": item["pred_label_name"],
                "pred_confidence": item["pred_confidence"],
                "primary_mode_hint": max(item["mode_scores"].items(), key=lambda kv: kv[1])[0],
                "file": f"analysis-4_cases/{item['example_id']}.json"
            })

    mode_summary = {}
    for j, mode in enumerate(mode_names):
        scores = mode_matrix[:, j]
        mode_summary[mode] = {
            "num_misclassified_scanned": int(len(candidates)),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std(ddof=0)),
            "p90_score": float(np.percentile(scores, 90)) if len(scores) else 0.0,
            "p99_score": float(np.percentile(scores, 99)) if len(scores) else 0.0,
        }

    proposed_fixes = {
        "A_context_poor_topk": [
            "Add stopword or low-information penalty in word scorer or add IDF prior features.",
            "Replace hard top-K with differentiable sparse attention like entmax or sparsemax.",
            "Use hybrid selection: top-K plus always keep rare or content tokens."
        ],
        "B_word_selection_mismatch": [
            "Add consistency loss aligning soft word attention and hard top-K selection.",
            "Use Gumbel-Softmax straight-through sampling for discrete selection alignment.",
            "Make K dynamic based on sentence length or attention entropy."
        ],
        "C_cross_attention_collapse": [
            "Add entropy regularization on sentence-importance distribution to avoid collapse.",
            "Use multi-head cross-attention and encourage head diversity.",
            "Add residual connection from global pooled doc vector to reduce brittle single-sentence reliance."
        ],
        "D_local_global_conflict": [
            "Add diagonal or locality bias so queries attend somewhat to origin sentence.",
            "Use sentence position embeddings or discourse features to stabilize alignment.",
            "Two-stage cross-attention: local neighborhood then global."
        ],
        "E_overconfident_error": [
            "Use temperature scaling for calibration after training.",
            "Use label smoothing or focal loss to reduce overconfident mistakes.",
            "Down-weight cross-attended features when attention entropy collapses."
        ]
    }

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
            "settings": {
                "TOP_CASES_PER_MODE": int(TOP_CASES_PER_MODE),
                "MAX_SCAN_EXAMPLES": MAX_SCAN_EXAMPLES
            },
            "files": {
                "cases_dir": "analysis-4_cases",
                "plots_dir": "analysis-4_plots"
            }
        },
        "failure_mode_definitions": {
            "A_context_poor_topk": "Top-K word selection dominated by stopwords or UNK or numbers, leading to low-information evidence passed to cross-attention.",
            "B_word_selection_mismatch": "High word-attention token is not selected in top-K, causing evidence loss before cross-attention.",
            "C_cross_attention_collapse": "Cross-attention concentrates on a single sentence, making decisions brittle.",
            "D_local_global_conflict": "Queries attend away from their origin sentence, causing misalignment of local and global signals.",
            "E_overconfident_error": "Wrong predictions with high confidence, indicating shortcut patterns and weak calibration."
        },
        "mode_score_summary_over_misclassifications": mode_summary,
        "selected_cases_index": case_index,
        "proposed_architectural_or_training_fixes": proposed_fixes
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
