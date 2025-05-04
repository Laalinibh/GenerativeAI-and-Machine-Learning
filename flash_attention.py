import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttentionMini(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size=32):
        super(FlashAttentionMini, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Initialize output
        output = torch.zeros_like(q)

        # Block processing
        for start in range(0, seq_len, self.block_size):
            end = min(start + self.block_size, seq_len)
            q_block = q[:, :, start:end, :]  # (batch, heads, block, head_dim)
            k_block = k[:, :, :end, :]        # (batch, heads, end, head_dim)
            v_block = v[:, :, :end, :]        # (batch, heads, end, head_dim)

            attn_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v_block)

            output[:, :, start:end, :] = attn_output

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(output)

# Example usage
x = torch.randn(2, 128, 512)  # (batch_size, seq_len, embed_dim)
attn = FlashAttentionMini(embed_dim=512, num_heads=8)
out = attn(x)
