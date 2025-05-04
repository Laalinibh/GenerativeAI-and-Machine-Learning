import torch
import torch.nn as nn
import torch.nn.functional as F

class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k=64, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # Shared projection matrix for keys and values
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads, self.head_dim).transpose(1, 2), qkv)

        # Project k and v using low-rank E: (B, heads, N, dim) -> (B, heads, k, dim)
        E_proj = self.E.to(x.device)
        k_proj = torch.matmul(k.transpose(2, 3), E_proj).transpose(2, 3)
        v_proj = torch.matmul(v.transpose(2, 3), E_proj).transpose(2, 3)

        attn_scores = torch.matmul(q, k_proj.transpose(-1, -2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v_proj)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

# Example usage
x = torch.randn(2, 128, 128)  # batch size=2, seq_len=128, model_dim=128
model = LinformerSelfAttention(dim=128, seq_len=128)
print(model(x).shape)  # Output: torch.Size([2, 128, 128])
