import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(MinimalLLM, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Embeddings and positional encodings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeddings = nn.Parameter(torch.randn(1, 512, embed_dim))

        # Transformer layers
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        
        # Output projection (logits)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, cache=None):
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        # Embeddings + positional encodings
        embeddings = self.token_embeddings(input_ids) + self.positional_embeddings[:, :seq_len, :]

        # Transformer layers with caching
        output = embeddings
        for layer in self.layers:
            output = layer(output)
        
        logits = self.output_proj(output)
        return logits

    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
        output = input_ids
        cache = None
        for _ in range(max_length):
            logits = self(output, cache=cache)
            logits = logits[:, -1, :] / temperature  # Only look at last token

            # Sampling using top-k
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Add the predicted token to the input sequence
            next_token = top_k_indices.gather(-1, next_token)
            output = torch.cat([output, next_token], dim=1)

        return output

# Example usage
vocab_size = 50257  # GPT-2 vocab size
embed_dim = 768
num_heads = 12
num_layers = 12

model = MinimalLLM(vocab_size, embed_dim, num_heads, num_layers)

# Input token (example)
input_ids = torch.tensor([[50256]])  # Start with a special token

# Generate text
generated_ids = model.generate(input_ids, max_length=20)

# Print generated tokens (usually you'd map these back to words)
print(generated_ids)
