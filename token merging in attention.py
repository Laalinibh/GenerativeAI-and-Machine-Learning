import torch
import torch.nn.functional as F

class TokenMergingAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, threshold=0.9):
        super(TokenMergingAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.threshold = threshold
        
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

        # Token similarity matrix
        sim_matrix = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Merge tokens based on similarity threshold
        merged_tokens = self.merge_tokens(sim_matrix, x)

        # Attention output
        attn_scores = F.softmax(sim_matrix, dim=-1)
        attn_output = torch.matmul(attn_scores, v)

        merged_output = torch.matmul(attn_scores, merged_tokens)  # Output with merged tokens
        output = self.out_proj(merged_output)
        
        return output

    def merge_tokens(self, sim_matrix, x):
        # Find token pairs that are very similar (above threshold)
        batch_size, num_heads, seq_len, head_dim = sim_matrix.shape
        mask = sim_matrix > self.threshold
        
        # Find indices of similar token pairs and merge them
        merged_tokens = []
        for i in range(seq_len):
            if mask[:, :, i].sum() > 0:  # Merge if enough similarity
                merged_tokens.append(torch.mean(x[:, i], dim=0))
        
        return torch.stack(merged_tokens, dim=1)  # Merge tokens

# Example usage
x = torch.randn(2, 128, 512)  # (batch_size, seq_len, embed_dim)
attn = TokenMergingAttention(embed_dim=512, num_heads=8)
out = attn(x)
