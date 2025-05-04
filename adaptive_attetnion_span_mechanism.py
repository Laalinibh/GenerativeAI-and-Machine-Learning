import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAttention(nn.Module):
    def __init__(self, dim, window_size=5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Default window size

        # Learnable attention span for each position
        self.attention_span = nn.Parameter(torch.randint(1, window_size + 1, (1,)))

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        output = torch.zeros_like(x)

        # Iterate over each token and apply local attention with adaptive span
        for t in range(T):
            span = min(self.attention_span.item(), t + 1)
            start = t - span + 1
            end = t + 1
            q_t = q[:, t:t+1, :]
            k_window = k[:, start:end, :]
            v_window = v[:, start:end, :]

            scores = torch.matmul(q_t, k_window.transpose(-1, -2)) / (D ** 0.5)
            weights = F.softmax(scores, dim=-1)
            attended = torch.matmul(weights, v_window)
            output[:, t:t+1, :] = attended

        return output

# Example usage
x = torch.randn(2, 10, 64)  # batch size=2, seq_len=10, model_dim=64
model = AdaptiveAttention(dim=64, window_size=5)
print(model(x).shape)  # Output: torch.Size([2, 10, 64])
