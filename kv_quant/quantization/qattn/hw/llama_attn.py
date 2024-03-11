"""
PyTorch LLaMA Attention model from llama: 
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.config.kv_bit is None or self.config.kv_bit >= 16:
            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None
        else:
            # TODO: quantization
            from ...quant_funcs_hw import pseudo_quantize_tensor
            from ...pack_funcs import write_8bit_tensor, write_4bit_tensor, write_3bit_tensor, write_2bit_tensor, write_1bit_tensor
            quant_key_layer, key_scale, key_zeros = pseudo_quantize_tensor(key_states, n_bits=self.config.kv_bit, q_group_size=self.config.kv_group_size)
            quant_value_layer, value_scale, value_zeros = pseudo_quantize_tensor(value_states, n_bits=self.config.kv_bit, q_group_size=self.config.kv_group_size)
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

            if past_key_value is not None:
                quant_key_layer = torch.cat((past_key_value[0], quant_key_layer) ,dim=2)
                quant_value_layer = torch.cat((past_key_value[1] ,quant_value_layer) ,dim=2)
                key_scale = torch.cat((self.key_scale, key_scale[-1, ...].unsqueeze(0)) ,dim=-1)
                key_zeros = torch.cat((self.key_zeros, key_zeros[-1, ...].unsqueeze(0)) ,dim=-1)
                value_scale = torch.cat((self.value_scale, value_scale[-1, ...].unsqueeze(0)) ,dim=1)
                value_zeros = torch.cat((self.value_zeros, value_zeros[-1, ...].unsqueeze(0)) ,dim=1)
            past_key_value = (quant_key_layer, quant_value_layer) if use_cache else None
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
            key_states = (int_key_layer - self.key_zeros.reshape(int_key_layer.shape[0], -1)) * self.key_scale.reshape(int_key_layer.shape[0], -1)
            value_states = (int_value_layer - self.value_zeros.reshape(int_value_layer.shape[0], -1)) * self.value_scale.reshape(int_value_layer.shape[0], -1)
            key_states = key_states.reshape(ori_key_shape)
            value_states = value_states.reshape(ori_value_shape)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
