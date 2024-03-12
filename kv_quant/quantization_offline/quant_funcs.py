import torch

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False, offline_cache=None, return_param=False):
    """
    The basic quantization function for weight, activation and KV cache.
    return_param: whether to return scales and zeros (for offline generating stage on calib dataset)
    offline_cache: whether to use offline-collected scales and zeros (for offline quantization using pre-generated scales and zeros)
    """
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        assert org_tensor_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    if not offline_cache:   # online quantization
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
    else:   # offline quantization, use pre-generated scales and zeros
        if zero_point:
            max_int = 2**n_bits - 1
            min_int = 0
            scales = offline_cache[0][:tensor.shape[0]].to(tensor.device)
            zeros = offline_cache[1][:tensor.shape[0]].to(tensor.device)
        else:
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -(2 ** (n_bits - 1))
            scales = offline_cache[0][:tensor.shape[0]].to(tensor.device)
            zeros = 0


    if inplace:
        (
            (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        tensor = (
            torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
        ) * scales

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    if return_param:
    # return the quantized tonsor, the scaling factor and the zero point value
        return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)    # if seq_len=13, then scales.shape and zeros.shape = [13,1]
    return tensor


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, offline_cache=None, return_param=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    res = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False, offline_cache=offline_cache, return_param=return_param)
    if return_param:
        tensor, scales, zeros = res
        return tensor, scales, zeros
    return res
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, offline_cache=None, return_param=False):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1]) # [1, 13, 4096] -> [13, 4096]
    res = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False, offline_cache=offline_cache, return_param=return_param)
    if return_param:
        tensor, scales, zeros = res # [13, 4096], [13, 1], [13, 1]
        return tensor.reshape(t_shape), scales, zeros   # [13, 4096] -> [1, 13, 4096]
    return res.reshape(t_shape)
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t = t.view(-1, t_shape[-1])
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=True, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape)
