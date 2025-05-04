import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryAugmentedAttention(nn.Module):
    def __init__(self, embed_dim, heads, memory_size):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # External memory: [memory_size x embed_dim]
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.heads, self.head_dim).transpose(1, 2), qkv)

        # Expand memory for batch
        mem_kv = self.memory.unsqueeze(0).expand(B, -1, -1)
        mem_k = mem_kv.clone()
        mem_v = mem_kv.clone()

        mem_k = mem_k.view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        mem_v = mem_v.view(B, -1, self.heads, self.head_dim).transpose(1, 2)

        # Concatenate memory with current keys and values
        k = torch.cat([mem_k, k], dim=2)
        v = torch.cat([mem_v, v], dim=2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

# Test the module
x = torch.randn(2, 32, 128)  # batch_size=2, seq_len=32, embed_dim=128
model = MemoryAugmentedAttention(embed_dim=128, heads=4, memory_size=16)
print(model(x).shape)  # Output: torch.Size([2, 32, 128])
