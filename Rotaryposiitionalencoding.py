import torch
import math

def apply_rotary_pos_emb(q, k, seq_dim=1):
    """
    Apply rotary positional embeddings to queries and keys.

    Args:
        q: (batch_size, seq_len, num_heads, head_dim)
        k: (batch_size, seq_len, num_heads, head_dim)
    Returns:
        Tuple of (q_rotated, k_rotated)
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    device = q.device

    # Make sure head_dim is even
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"

    # Create position indices
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Compute frequencies
    dim = torch.arange(0, head_dim, 2, device=device)
    inv_freq = 1.0 / (10000 ** (dim / head_dim))

    # Position * Frequencies
    sinusoid_inp = torch.einsum('i,j->ij', pos, inv_freq)

    # Compute sin and cos
    sin = sinusoid_inp.sin()[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
    cos = sinusoid_inp.cos()[None, :, None, :]

    # Split into pairs
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # Apply rotation
    q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rotated, k_rotated

# Example usage
q = torch.randn(2, 5, 8, 64)  # (batch_size, seq_len, num_heads, head_dim)
k = torch.randn(2, 5, 8, 64)
q_rot, k_rot = apply_rotary_pos_emb(q, k)
