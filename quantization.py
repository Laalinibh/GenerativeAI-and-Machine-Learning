import torch

def quantize_dequantize(tensor, bits=8):
    qmin = 0
    qmax = 2**bits - 1

    min_val = tensor.min()
    max_val = tensor.max()

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    # Quantize
    q_tensor = zero_point + tensor / scale
    q_tensor.clamp_(qmin, qmax).round_()

    # Dequantize
    dq_tensor = (q_tensor - zero_point) * scale
    return dq_tensor

# Example usage
x = torch.randn(4, 4)
x_dq = quantize_dequantize(x)
