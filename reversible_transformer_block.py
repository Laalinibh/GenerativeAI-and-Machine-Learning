import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class ReversibleBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.f = FeedForward(dim, hidden_dim)
        self.g = FeedForward(dim, hidden_dim)

    def forward(self, x1, x2):
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return y1, y2

    def backward_pass(self, y1, y2):
        # Reverse operations to save memory
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return x1, x2

# Test reversible forward and reverse
x1 = torch.randn(4, 128)
x2 = torch.randn(4, 128)
block = ReversibleBlock(dim=128, hidden_dim=256)

y1, y2 = block(x1, x2)
recovered_x1, recovered_x2 = block.backward_pass(y1, y2)

# Check recovery
print(torch.allclose(x1, recovered_x1, atol=1e-5))  # Should be True
print(torch.allclose(x2, recovered_x2, atol=1e-5))  # Should be True
