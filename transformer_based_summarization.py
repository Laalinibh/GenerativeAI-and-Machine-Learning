import torch
import torch.nn as nn
from torch.nn import Transformer

class SummarizationTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.transformer = Transformer(d_model=d_model, nhead=nhead,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers,
                                       dim_feedforward=dim_feedforward)

        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_embed = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        tgt_embed = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]

        src_embed = src_embed.transpose(0, 1)  # (seq_len, batch, d_model)
        tgt_embed = tgt_embed.transpose(0, 1)

        memory = self.transformer(src_embed, tgt_embed)
        out = self.generator(memory.transpose(0, 1))  # (batch, seq_len, vocab_size)
        return out

# Dummy input: vocabulary size = 10000
src = torch.randint(0, 10000, (2, 64))  # batch_size=2, src_len=64
tgt = torch.randint(0, 10000, (2, 32))  # batch_size=2, tgt_len=32

model = SummarizationTransformer(vocab_size=10000)
out = model(src, tgt)
print(out.shape)  # Output: torch.Size([2, 32, 10000])
