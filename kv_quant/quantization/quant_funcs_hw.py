import torch

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0

    tensor = torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int)

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    # return the quantized tonsor, the scaling factor and the zero point value
    return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
