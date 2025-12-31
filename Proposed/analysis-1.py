import os
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
EX_DIR = os.path.join(REPORT_DIR, "analysis-1_examples")
PLOTS_DIR = os.path.join(REPORT_DIR, "analysis-1_plots")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(EX_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def decode_tokens(token_ids_1d, itos, pad_id):
    toks = []
    for tid in token_ids_1d.tolist():
        if tid == pad_id:
            break
        toks.append(itos[tid] if 0 <= tid < len(itos) else "<UNK>")
    return toks


def safe_float(x):
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


def topk_from_vector(values, k):
    k = min(k, values.numel())
    vals, idx = torch.topk(values, k)
    return idx.tolist(), vals.tolist()


def save_sentence_importance_bar(example_id, ranked, out_dir):
    ranked = sorted(ranked, key=lambda x: x["sentence_id"])
    xs = [str(r["sentence_id"]) for r in ranked]
    ys = [r["importance"] for r in ranked]

    plt.figure(figsize=(8, 3))
    plt.bar(xs, ys)
    plt.xlabel("Sentence id")
    plt.ylabel("Importance")
    plt.title(f"Sentence importance — {example_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{example_id}_sentence_importance.png"), dpi=200)
    plt.close()


def save_cross_attention_heatmap(example_id, query_summaries, attn, ns, out_dir, max_q=25):
    qn = min(attn.shape[0], max_q)
    mat = attn[:qn, :ns].cpu().numpy()

    ylabels = []
    for q in range(qn):
        qs = query_summaries[q]
        ylabels.append(f"s{qs['from_sentence_id']}:{qs['token']}")

    plt.figure(figsize=(10, max(3, 0.3 * qn)))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Cross-attn")
    plt.xticks(range(ns), [f"s{i}" for i in range(ns)])
    plt.yticks(range(qn), ylabels)
    plt.xlabel("Sentence")
    plt.ylabel("Query word")
    plt.title(f"Cross-attention heatmap — {example_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{example_id}_cross_attention.png"), dpi=200)
    plt.close()


def save_word_topk_bars(example_id, sentence_summaries, out_dir, max_sents=6):
    for ss in sentence_summaries[:max_sents]:
        tokens = [w["token"] for w in ss["top_words"]]
        weights = [w["word_attn_weight"] for w in ss["top_words"]]

        plt.figure(figsize=(8, 3))
        plt.bar(tokens, weights)
        plt.ylabel("Word attention")
        plt.title(f"Top-K words — {example_id} — sentence {ss['sentence_id']}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{example_id}_word_topk_s{ss['sentence_id']}.png"),
            dpi=200
        )
        plt.close()


@torch.no_grad()
def extract_one_example(model, batch, itos, pad_id, target_names, K):
    x, y, sent_lengths, num_sents = batch
    x, y = x.to(config.device), y.to(config.device)
    sent_lengths, num_sents = sent_lengths.to(config.device), num_sents.to(config.device)

    dbg = model(x, sent_lengths, num_sents, return_debug=True)

    probs = F.softmax(dbg["logits"], dim=-1)
    preds = probs.argmax(dim=-1)

    results = []
    B, S, T = x.shape

    for b in range(B):
        ns = int(num_sents[b])
        true_id = int(y[b])
        pred_id = int(preds[b])
        conf = float(probs[b, pred_id])

        decoded = [decode_tokens(x[b, s], itos, pad_id) for s in range(S)]

        # Sentence summaries
        sent_summaries = []
        for s in range(ns):
            items = []
            for r, pos in enumerate(dbg["topk_idx"][b, s].tolist()):
                tok = decoded[s][pos] if pos < len(decoded[s]) else "<PAD>"
                items.append({
                    "rank": r,
                    "token": tok,
                    "token_pos": pos,
                    "word_score_logit": safe_float(dbg["scores"][b, s, pos]),
                    "word_attn_weight": safe_float(dbg["attn_words"][b, s, pos])
                })
            sent_summaries.append({
                "sentence_id": s,
                "sentence_text": " ".join(decoded[s]),
                "top_words": items
            })

        # Query summaries
        query_summaries = []
        doc_attn = dbg["doc_attn_weights"][b]
        for s in range(ns):
            for k in range(K):
                q = s * K + k
                pos = dbg["topk_idx"][b, s, k].item()
                tok = decoded[s][pos] if pos < len(decoded[s]) else "<PAD>"
                attn_row = doc_attn[q, :ns]

                top_ids, top_vals = topk_from_vector(attn_row, 3)
                links = [{
                    "rank": i,
                    "sentence_id": sid,
                    "attn_weight": safe_float(val),
                    "sentence_text": " ".join(decoded[sid])
                } for i, (sid, val) in enumerate(zip(top_ids, top_vals))]

                query_summaries.append({
                    "query_id": q,
                    "from_sentence_id": s,
                    "token": tok,
                    "token_pos": int(pos),
                    "word_score_logit": safe_float(dbg["scores"][b, s, pos]),
                    "word_attn_weight": safe_float(dbg["attn_words"][b, s, pos]),
                    "top_attended_sentences": links
                })

        qn = ns * K
        agg = doc_attn[:qn, :ns].mean(dim=0)
        agg = (agg / agg.sum()).cpu().tolist()

        ranked = sorted(
            [{"sentence_id": i, "importance": agg[i]} for i in range(ns)],
            key=lambda x: x["importance"],
            reverse=True
        )

        out = {
            "true_label_id": true_id,
            "pred_label_id": pred_id,
            "true_label_name": target_names[true_id],
            "pred_label_name": target_names[pred_id],
            "pred_confidence": conf,
            "num_sents": ns,
            "sentence_summaries": sent_summaries,
            "query_summaries": query_summaries,
            "sentence_importance_from_cross_attn": {
                "definition": "mean cross-attn over queries",
                "values": [{"sentence_id": i, "importance": agg[i]} for i in range(ns)],
                "ranked": ranked
            },
            "_plot_payload": {
                "doc_attn": doc_attn[:qn, :ns].detach()
            }
        }
        results.append(out)

    return results



example_counter = 1

def main():
    global example_counter
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    Xtr, Xv, Xt, ytr, yv, yt, target_names, _ = load_splits(
        config.REMOVE_PARTS, config.SEED, config.VAL_SPLIT
    )

    stoi, itos, pad_id, unk_id, _ = build_vocab(Xtr, config.MIN_FREQ, config.MAX_VOCAB)
    glove = load_glove(config.GLOVE_PATH, config.EMBED_DIM)
    emb, coverage = build_embedding_matrix(stoi, pad_id, glove, config.EMBED_DIM)

    _, _, test_loader = make_loaders_sentence(
        Xtr, ytr, Xv, yv, Xt, yt,
        stoi, unk_id, pad_id,
        config.MAX_SENTS, config.MAX_TOKENS_PER_SENT,
        config.BATCH_SIZE
    )

    TOPK = 5
    model = BiLSTMWithGlove(
        len(stoi), config.EMBED_DIM, config.HIDDEN_DIM,
        len(target_names), pad_id, emb,
        config.DROPOUT, config.FREEZE_EMBEDDINGS,
        topk_words=TOPK, attn_dropout=0.1
    ).to(config.device)

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.device))
    model.eval()

    analysis_idx = next_analysis_index()
    meta = {
        "analysis_id": f"analysis-{analysis_idx}",
        "created_at": datetime.now().isoformat(),
        "split": "test",
        "glove_coverage": float(coverage),
        "TOPK": TOPK
    }

    examples_index = []
    saved = 0
    MAX_EX = 50

    for bi, batch in enumerate(test_loader):
        outs = extract_one_example(model, batch, itos, pad_id, target_names, TOPK)

        for ei, e in enumerate(outs):
            ex_id = f"test-{example_counter}"
            e["example_id"] = ex_id

            # Save JSON
            with open(os.path.join(EX_DIR, f"{ex_id}.json"), "w") as f:
                json.dump({k: v for k, v in e.items() if k != "_plot_payload"}, f, indent=2)

            # Save plots
            save_sentence_importance_bar(
                ex_id, e["sentence_importance_from_cross_attn"]["ranked"], PLOTS_DIR
            )
            save_cross_attention_heatmap(
                ex_id, e["query_summaries"],
                e["_plot_payload"]["doc_attn"],
                e["num_sents"], PLOTS_DIR
            )
            save_word_topk_bars(ex_id, e["sentence_summaries"], PLOTS_DIR)

            examples_index.append({
                "example_id": ex_id,
                "true": e["true_label_name"],
                "pred": e["pred_label_name"],
                "confidence": e["pred_confidence"]
            })

            saved += 1

            example_counter += 1   # ✅ critical
            saved += 1
            if saved >= MAX_EX:
                break
        if saved >= MAX_EX:
            break

    with open(os.path.join(REPORT_DIR, f"analysis-{analysis_idx}.json"), "w") as f:
        json.dump({"meta": meta, "examples": examples_index}, f, indent=2)

if __name__ == "__main__":
    main()
