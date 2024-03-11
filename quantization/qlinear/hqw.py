import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..pack_funcs import *
from ..quant_funcs_hw import pseudo_quantize_tensor

class WLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, dev, w_bit, weight_group):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.weight_group = weight_group
        assert self.in_features % self.weight_group == 0

        self.register_buffer('qweight', torch.zeros((out_features, in_features * w_bit // 8), dtype=torch.uint8, device=dev))
        self.register_buffer('weight_scale', torch.zeros((out_features, in_features // self.weight_group), dtype=torch.float16, device=dev))
        self.register_buffer('weight_zero', torch.zeros((out_features, in_features // self.weight_group), dtype=torch.int8, device=dev))
        if bias:
            # fp16 bias
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear, w_bit, weight_group, init_only=False):
        qlinear = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, w_bit, weight_group)
        if init_only:
            return qlinear
        
        # store bias
        if linear.bias is not None:
            qlinear.bias = linear.bias.clone().half()
        
        q_weight, weight_scale, weight_zero = pseudo_quantize_tensor(linear.weight, n_bits=w_bit, q_group_size=weight_group)
        qlinear.weight_scale = weight_scale.to(linear.weight.device).to(linear.weight.dtype)
        qlinear.weight_zero = weight_zero.to(linear.weight.device).to(torch.int8)
        
        # quantize weight
        if w_bit == 8:
            qlinear.qweight = write_8bit_tensor(q_weight)
        elif w_bit == 4:
            qlinear.qweight = write_4bit_tensor(q_weight)
        elif w_bit == 3:
            qlinear.qweight = write_3bit_tensor(q_weight)
        elif w_bit == 2:
            qlinear.qweight = write_2bit_tensor(q_weight)
        elif w_bit == 1:
            qlinear.qweight = write_1bit_tensor(q_weight)
        else:
            raise NotImplementedError(f"Not support {w_bit} bit quantization")
        
        return qlinear
    
    def gemm_forward_cuda(self, x):
        if self.w_bit == 8:
            w_int8 = deq_8bit_tensor(self.qweight).to(torch.float16)
        elif self.w_bit == 4:
            w_int8 = deq_4bit_tensor(self.qweight).to(torch.float16)
        elif self.w_bit == 3:
            w_int8 = deq_3bit_tensor(self.qweight).to(torch.float16)
        elif self.w_bit == 2:
            w_int8 = deq_2bit_tensor(self.qweight).to(torch.float16)
        elif self.w_bit == 1:
            w_int8 = deq_1bit_tensor(self.qweight).to(torch.float16)
        else:
            raise NotImplementedError(f"Not support {self.w_bit} bit quantization")
        
        w_int8_org_shape = w_int8.shape
        w_int8 = w_int8.reshape(-1, self.weight_group)
        w_int8 = w_int8 - self.weight_zero.reshape(-1,1)

        w = w_int8 * self.weight_scale.reshape(-1,1)
        w = w.reshape(w_int8_org_shape).to(torch.float16)

        return F.linear(x, w, self.bias)

    @torch.no_grad()
    def forward(self, x):
        out = self.gemm_forward_cuda(x)
        return out