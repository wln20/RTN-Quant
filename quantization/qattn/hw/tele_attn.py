# coding=utf-8
"""PyTorch TELECHAT model."""
from typing import Optional, Tuple

import torch
import math
from torch import nn
import torch.utils.checkpoint
from torch.nn import functional as F

try:
    from einops import rearrange
except ImportError:
    rearrange = None

use_flash_attn = True
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None



class RotaryEmbedding(torch.nn.Module):
    # Extracted from: https://github.com/EleutherAI/gpt-neox
    def __init__(self, dim ,config, base=10000, precision=torch.half):
        super().__init__()
        self.config = config
        self.dim = dim
        self.base = base
        self.inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().half() / dim)).cuda()
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def get_mscale(self,scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def get_ntk_alpha(self, true_seq_len):
        context_value = math.log(true_seq_len / 4096, 2) + 1
        # ntk_alpha = 2 ** context_value - 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def forward(self, x, seq_dim=0, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        seq_len = max(seq_len, self.config.training_seqlen)
        ntk_alpha = self.get_ntk_alpha(seq_len)
        self.mscale = float(self.get_mscale(seq_len / self.config.training_seqlen))
        if True:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float( )/ self.dim ))
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            self.cos_cached = self.mscale *emb.cos()[:, None, :].half()
            self.sin_cached = self.mscale *emb.sin()[:, None, :].half()
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]



# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions

def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):  # jitting fails with bf16
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class FlashSelfAttention(torch.nn.Module):
    # Extracted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/model/transformer.py
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
        self.training = False
        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                        device=q.device)
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class TelechatAttention(nn.Module):
    def __init__(self, config ,layer_idx):
        super().__init__()
        self.kv_cache = None
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.num_key_value_heads = self.num_heads
        kv_projection_size = self.head_dim * self.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_value = nn.Linear(self.hidden_size, kv_projection_size * 2, bias=False)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim ,config=config)

        self.core_attention_flash = FlashSelfAttention(
            causal=True, attention_dropout=config.attention_dropout
        )

        self.last_key_layer = None
        logn_list = [math.log(i, 4096) if i > 4096 else 1 for i in range(1, 32768)]
        self.logn_tensor = torch.tensor(logn_list)[None, :, None, None].half().cuda()


    def repeat_kv(self, hidden_states, n_rep):
        slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(slen, batch, num_key_value_heads_per_partition, n_rep,
                                                               head_dim)
        return hidden_states.reshape(slen, batch, num_key_value_heads_per_partition * n_rep, head_dim)

    def split_tensor_along_last_dim(self,
                                    tensor: torch.Tensor,
                                    num_partitions: int,
                                    contiguous_split_chunks: bool = False,
                                    ):

        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        hidden_states = hidden_states.transpose(1, 0)
        query_layer = self.query(hidden_states)
        new_tensor_shape = query_layer.size()[:-1] + \
                           (self.num_heads,
                            self.head_dim)
        query_layer = query_layer.view(*new_tensor_shape)

        mixed_kv_layer = self.key_value(hidden_states)
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                           (self.num_key_value_heads,
                            2 * self.head_dim)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
        (key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_kv_layer, 2)

        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        apply_rotary_fn = apply_rotary_pos_emb_torch

        seq_len = key_layer.shape[0]
        offset = 0

        if use_cache and layer_past != None:
            past_key, past_value  = layer_past
            offset = past_key.shape[0]
            seq_len += offset

        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)

        query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset)

        if use_cache:
            if self.config.kv_bit is None or self.config.kv_bit >= 16:
                if layer_past != None:
                    past_key, past_value = layer_past
                    key_layer = torch.cat((past_key, key_layer[-1, ...].unsqueeze(0)) ,dim=0)
                    value_layer = torch.cat((past_value ,value_layer[-1 ,...].unsqueeze(0)) ,dim = 0)
                layer_past = key_layer ,value_layer
            else:
                # TODO: real quantization
                from ...quant_funcs_hw import pseudo_quantize_tensor
                from ...pack_funcs import write_8bit_tensor, write_4bit_tensor, write_3bit_tensor, write_2bit_tensor, write_1bit_tensor
                quant_key_layer, key_scale, key_zeros = pseudo_quantize_tensor(key_layer, n_bits=self.config.kv_bit, q_group_size=self.config.kv_group_size)
                quant_value_layer, value_scale, value_zeros = pseudo_quantize_tensor(value_layer, n_bits=self.config.kv_bit, q_group_size=self.config.kv_group_size)
                if self.config.kv_bit == 8:
                    quant_key_layer = write_8bit_tensor(quant_key_layer)
                    quant_value_layer = write_8bit_tensor(quant_value_layer)
                elif self.config.kv_bit == 4:
                    quant_key_layer = write_4bit_tensor(quant_key_layer)
                    quant_value_layer = write_4bit_tensor(quant_value_layer)
                elif self.config.kv_bit == 3:
                    quant_key_layer = write_3bit_tensor(quant_key_layer)
                    quant_value_layer = write_3bit_tensor(quant_value_layer)
                elif self.config.kv_bit == 2:
                    quant_key_layer = write_2bit_tensor(quant_key_layer)
                    quant_value_layer = write_2bit_tensor(quant_value_layer)
                elif self.config.kv_bit == 1:
                    quant_key_layer = write_1bit_tensor(quant_key_layer)
                    quant_value_layer = write_1bit_tensor(quant_value_layer)
                else:
                    raise NotImplementedError(f"Not support {self.config.kv_bit} bit quantization")
                
                if layer_past != None:
                    past_quant_key, past_quant_value = layer_past
                    quant_key_layer = torch.cat((past_quant_key, quant_key_layer[-1, ...].unsqueeze(0)) ,dim=0)
                    quant_value_layer = torch.cat((past_quant_value ,quant_value_layer[-1 ,...].unsqueeze(0)) ,dim = 0)
                    key_scale = torch.cat((self.key_scale, key_scale[-1, ...].unsqueeze(0)) ,dim=0)
                    key_zeros = torch.cat((self.key_zeros, key_zeros[-1, ...].unsqueeze(0)) ,dim=0)
                    value_scale = torch.cat((self.value_scale, value_scale[-1, ...].unsqueeze(0)) ,dim=0)
                    value_zeros = torch.cat((self.value_zeros, value_zeros[-1, ...].unsqueeze(0)) ,dim=0)
                layer_past = quant_key_layer ,quant_value_layer
                self.key_scale = key_scale
                self.key_zeros = key_zeros
                self.value_scale = value_scale
                self.value_zeros = value_zeros
                
                from ...pack_funcs import deq_8bit_tensor, deq_4bit_tensor, deq_3bit_tensor, deq_2bit_tensor, deq_1bit_tensor
                if self.config.kv_bit == 8:
                    int_key_layer = deq_8bit_tensor(quant_key_layer)
                    int_value_layer = deq_8bit_tensor(quant_value_layer)
                elif self.config.kv_bit == 4:
                    int_key_layer = deq_4bit_tensor(quant_key_layer)
                    int_value_layer = deq_4bit_tensor(quant_value_layer)
                elif self.config.kv_bit == 3:
                    int_key_layer = deq_3bit_tensor(quant_key_layer)
                    int_value_layer = deq_3bit_tensor(quant_value_layer)
                elif self.config.kv_bit == 2:
                    int_key_layer = deq_2bit_tensor(quant_key_layer)
                    int_value_layer = deq_2bit_tensor(quant_value_layer)
                elif self.config.kv_bit == 1:
                    int_key_layer = deq_1bit_tensor(quant_key_layer)
                    int_value_layer = deq_1bit_tensor(quant_value_layer)
                else:
                    raise NotImplementedError(f"Not support {self.config.kv_bit} bit quantization")
                
                int_key_layer = int_key_layer.to(torch.float16)
                int_value_layer = int_value_layer.to(torch.float16)

                # minus zeros and multiply scales
                ori_key_shape = int_key_layer.shape
                ori_value_shape = int_value_layer.shape
                int_key_layer = int_key_layer.reshape(-1, self.config.kv_group_size)
                int_value_layer = int_value_layer.reshape(-1, self.config.kv_group_size)
                key_layer = (int_key_layer - self.key_zeros.reshape(int_key_layer.shape[0], -1)) * self.key_scale.reshape(int_key_layer.shape[0], -1)
                value_layer = (int_value_layer - self.value_zeros.reshape(int_value_layer.shape[0], -1)) * self.value_scale.reshape(int_value_layer.shape[0], -1)
                key_layer = key_layer.reshape(ori_key_shape)
                value_layer = value_layer.reshape(ori_value_shape)

        s, bz, head, dim = value_layer.shape
        s_key = key_layer.shape[0]
        s_query = query_layer.shape[0]
        query_layer = query_layer.reshape((s_query, bz, head, dim))
        key_layer = key_layer.reshape((s_key, bz, head, dim))


        if self.config.flash_attn:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous() for x in
                       (query_layer, key_layer, value_layer)]
            if self.config.logn:
                seq_start = key_layer.size(1) - query_layer.size(1)
                seq_end = key_layer.size(1)
                logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
                q = q * logn_tensor.expand_as(q)
            context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> b s (h d)').contiguous()
        else:
            if self.config.logn:
                q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous() for x in (query_layer, key_layer, value_layer)]
                seq_start = key_layer.size(1) - query_layer.size(1)
                seq_end = key_layer.size(1)
                logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
                q = q * logn_tensor.expand_as(q)
                query_layer = rearrange(q, 'b s ... -> s b ...').contiguous()
            ##[sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.reshape(s_query ,bz * self.num_heads, dim)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.reshape(s_key, bz * self.num_heads, dim)
            matmul_result = self.inv_norm_factor * torch.einsum('bik,bkj->bij', query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2))

            attention_scores = matmul_result.view(bz, self.num_heads, s_query, s_key)

            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16:
                attention_scores = attention_scores.to(torch.float)
            attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
            attention_probs = F.softmax(attn_weights, dim=-1).to(input_dtype)  ##dtype = torch.float32
            attention_probs = self.attention_dropout(attention_probs)
            attention_probs_reshaped = attention_probs.view(bz * self.num_heads, s_query, s_key)

            value_layer = value_layer.reshape(s_key ,bz * self.num_heads, dim)
            context_layer = torch.bmm(attention_probs_reshaped, value_layer.transpose(0, 1))
            context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        present = None
        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return output_tensor, layer_past
