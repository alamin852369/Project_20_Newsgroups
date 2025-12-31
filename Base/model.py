
import torch
import torch.nn as nn

class BiLSTMWithGlove(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_id,
                 embedding_matrix, dropout=0.5, freeze=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths, return_sequence=False):

        emb = self.dropout(self.embedding(x))

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, (h, c) = self.lstm(packed)


        word_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        print("word_out", word_out.shape)

        mask = (x != self.embedding.padding_idx).float().unsqueeze(-1)
        summed = (word_out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        sent_vec = summed / denom
        print("sent_vec", sent_vec.shape)

        logits = self.fc(self.dropout(sent_vec))

        if return_sequence:
            return logits, word_out
        return logits

