import torch, math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=2, num_classes=4,
                 max_len=128, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len, dropout)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        pad_mask = (x == 0)
        x = self.pos_enc(self.embedding(x))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask   = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.fc(self.dropout(pooled))