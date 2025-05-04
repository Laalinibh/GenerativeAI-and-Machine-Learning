import torch
import torch.nn as nn

class SimpleDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, past_kv=None):
        """
        x: (seq_len, batch_size, embed_dim)
        past_kv: dict with keys 'k' and 'v', each (past_seq_len, batch_size, embed_dim)
        """
        if past_kv:
            past_k, past_v = past_kv['k'], past_kv['v']
            k = torch.cat([past_k, x], dim=0)
            v = torch.cat([past_v, x], dim=0)
        else:
            k = v = x

        attn_output, _ = self.self_attn(x, k, v, need_weights=False)
        output = self.ffn(attn_output)
        new_past_kv = {'k': k, 'v': v}
        return output, new_past_kv
