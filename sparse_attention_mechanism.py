import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseSelfAttention(nn.Module):
    def __init__(self, embed_dim, block_size=4, num_global_tokens=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, D = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_output = torch.zeros_like(x)

        for i in range(0, T, self.block_size):
            end = min(i + self.block_size, T)
            block_q = q[:, i:end, :]  # (B, block, D)

            # Local window: attend to nearby blocks (left, center, right)
            left = max(i - self.block_size, 0)
            right = min(i + 2 * self.block_size, T)
            block_k = k[:, left:right, :]
            block_v = v[:, left:right, :]

            scores = torch.matmul(block_q, block_k.transpose(-1, -2)) / D**0.5
            weights = F.softmax(scores, dim=-1)
            attended = torch.matmul(weights, block_v)
            attn_output[:, i:end, :] = attended

        # Optional: add global attention to specific tokens (e.g., first N tokens)
        global_q = q[:, :self.num_global_tokens, :]
        global_k = k
        global_v = v
        global_scores = torch.matmul(global_q, global_k.transpose(-1, -2)) / D**0.5
        global_weights = F.softmax(global_scores, dim=-1)
        global_out = torch.matmul(global_weights, global_v)
        attn_output[:, :self.num_global_tokens, :] = global_out

        return self.out_proj(attn_output)

# Example input
x = torch.randn(2, 16, 64)  # batch_size=2, seq_len=16, embed_dim=64
model = SparseSelfAttention(embed_dim=64, block_size=4, num_global_tokens=2)
print(model(x).shape)  # Output: torch.Size([2, 16, 64])
