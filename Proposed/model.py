import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentCrossAttentionMHA(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, top_word_vecs, sent_vecs, num_sents=None,
                return_doc_attended=False, return_attn=False):
        B, S, K, D = top_word_vecs.shape
        if D != self.dim:
            raise ValueError(f"Expected D={self.dim}, got {D}")
        if sent_vecs.shape != (B, S, D):
            raise ValueError(f"Expected sent_vecs {(B,S,D)}, got {tuple(sent_vecs.shape)}")

        Q = top_word_vecs.reshape(B, S * K, D)
        K_in = sent_vecs
        V_in = sent_vecs

        key_padding_mask = None
        if num_sents is not None:
            idx = torch.arange(S, device=sent_vecs.device).unsqueeze(0)
            key_padding_mask = idx >= num_sents.unsqueeze(1)

        attn_out, attn_w = self.mha(
            query=Q,
            key=K_in,
            value=V_in,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )

        doc_attended = self.norm(Q + self.dropout(attn_out))

        doc_vec = doc_attended.mean(dim=1)

        outs = [doc_vec]
        if return_doc_attended:
            outs.append(doc_attended)
        if return_attn:
            outs.append(attn_w)
        return outs[0] if len(outs) == 1 else tuple(outs)


class WordFilterSelfAttention(nn.Module):
    def __init__(self, dim, pad_id, topk=5, use_soft_weights=False):
        super().__init__()
        self.pad_id = pad_id
        self.topk = topk
        self.use_soft_weights = use_soft_weights

        self.scorer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, word_out, x):
        B, S, T, D = word_out.shape
        pad_mask = (x == self.pad_id)

        scores = self.scorer(word_out).squeeze(-1)
        scores = scores.masked_fill(pad_mask, -1e9)
        attn = F.softmax(scores, dim=-1)

        k = min(self.topk, T)
        topk_idx = torch.topk(scores, k=k, dim=-1).indices

        keep_mask = torch.zeros((B, S, T), device=word_out.device, dtype=torch.float32)
        keep_mask.scatter_(-1, topk_idx, 1.0)
        keep_mask = keep_mask.masked_fill(pad_mask, 0.0)

        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        top_word_vecs = torch.gather(word_out, dim=2, index=idx_expanded)

        if self.use_soft_weights:
            filtered_word_out = word_out * attn.unsqueeze(-1)
            return filtered_word_out, scores, keep_mask, attn, topk_idx, top_word_vecs

        km = keep_mask.unsqueeze(-1)
        filtered_word_out = word_out * km + word_out.detach() * (1.0 - km)
        return filtered_word_out, scores, keep_mask, attn, topk_idx, top_word_vecs


class BiLSTMWithGlove(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_id,
                 embedding_matrix, dropout=0.5, freeze=False,
                 topk_words=5, attn_dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.word_filter = WordFilterSelfAttention(dim=2 * hidden_dim, pad_id=pad_id, topk=topk_words)
        self.doc_attn = DocumentCrossAttentionMHA(
            dim=2 * hidden_dim,
            num_heads=8,
            dropout=attn_dropout
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, sent_lengths, num_sents=None, return_debug=False):
        B, S, T = x.shape

        x2 = x.view(B * S, T)
        lengths2 = sent_lengths.view(B * S)

        valid = lengths2 > 0
        if valid.sum().item() == 0:
            raise ValueError("All sentences have zero length in this batch.")

        x2_valid = x2[valid]
        lengths_valid = lengths2[valid]

        emb = self.dropout(self.embedding(x2_valid))

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_valid.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)

        out_valid, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )

        out_all = torch.zeros((B * S, T, 2 * self.hidden_dim), device=x.device, dtype=out_valid.dtype)
        out_all[valid] = out_valid
        word_out = out_all.view(B, S, T, 2 * self.hidden_dim)

        # print("word_out: ", word_out.shape)

        mask_words = (x != self.pad_id).float().unsqueeze(-1)
        sent_sum = (word_out * mask_words).sum(dim=2)
        # print("sent_sum: ", sent_sum.shape)
        sent_den = mask_words.sum(dim=2).clamp(min=1e-6)
        sent_vecs = sent_sum / sent_den
        # print("sent_vecs: ", sent_vecs.shape)

        if num_sents is None:
            num_sents = (sent_lengths > 0).sum(dim=1)

        # Word filter + topK
        filtered_word_out, scores, keep_mask, attn_words, topk_idx, top_word_vecs = self.word_filter(word_out, x)

        # Cross-attention
        doc_vec, doc_attended, doc_attn_weights = self.doc_attn(
            top_word_vecs,
            sent_vecs,
            num_sents=num_sents,
            return_doc_attended=True,
            return_attn=True
        )

        logits = self.fc(self.dropout(doc_vec))

        if not return_debug:
            return logits

        return {
            "logits": logits,
            "word_out": word_out,
            "sent_vecs": sent_vecs,

            # word attention
            "scores": scores,
            "attn_words": attn_words,
            "keep_mask": keep_mask,
            "topk_idx": topk_idx,
            "top_word_vecs": top_word_vecs,

            # cross attention
            "doc_vec": doc_vec,
            "doc_attended": doc_attended,
            "doc_attn_weights": doc_attn_weights,
            "num_sents": num_sents
        }

