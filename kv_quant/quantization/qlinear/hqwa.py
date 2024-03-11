import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from ..pack_funcs import *
from ..quant_funcs_hw import pseudo_quantize_tensor

@torch.no_grad()
def quantize_activation_per_token_absmax(t, scales, n_bits=8):
    token_len = t.shape[1]
    q_max = 2**(n_bits-1)-1
    t.div_(scales).round_().clamp_(-q_max, q_max).mul_(scales)
    return t

class WALinear(nn.Module):
    def __init__(self, in_features, out_features, bias, dev, w_bit, a_bit, weight_group, max_length):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.weight_group = weight_group if weight_group != -1 else in_features
        self.length = 0
        
        assert self.in_features % self.group_size == 0

        # W8 signed weight
        self.register_buffer('qweight', torch.zeros((out_features, in_features * w_bit // 8), dtype=torch.uint8, device=dev))

        # scales
        self.register_buffer('act_scale', torch.zeros((1, max_length, 1), dtype=torch.float16, device=dev))
        self.register_buffer('weight_scale', torch.zeros((out_features, in_features // self.weight_group), dtype=torch.float16, device=dev))
        self.register_buffer('weight_zero', torch.zeros((out_features, in_features // self.weight_group), dtype=torch.int8, device=dev))

        if bias:
            # fp16 bias
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear, weight_quant, act_quant, w_bit=4, a_bit=8, weight_group=128, max_length=2048, act_scale=None, init_only=False):
        qlinear = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, w_bit, a_bit, weight_group, max_length)
        if init_only:
            return qlinear
        
        # store bias
        if linear.bias is not None:
            qlinear.bias = linear.bias.clone().half()

        # store scales
        qlinear.act_scale = act_scale.to(linear.weight.device).to(linear.weight.dtype)

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
        w_int8 = w_int8.reshape(-1, self.group_size)
        w_int8 = w_int8 - self.weight_zero.reshape(-1,1)

        w = w_int8 * self.weight_scale.reshape(-1,1)
        w = w.reshape(w_int8_org_shape).to(torch.float16)

        # start compute, each iter we select one 128 input channel for computation
        if x.shape[1] == 1:
            act_scale_temp = self.act_scale[:,self.length:self.length+1,:]
            self.length += 1
        else:
            act_scale_temp = self.act_scale[:,:x.shape[1],:]
            self.length = x.shape[1]

        act_scale_temp = act_scale_temp.to(x.device)
        x_int = quantize_activation_per_token_absmax(x, act_scale_temp).to(torch.float16)

        out = F.linear(x_int, w, bias=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    @torch.no_grad()
    def forward(self, x):
        flag_0 = False
        flag_1 = False
        if x.dim() == 2:
            if x.shape[0] == 1:
                flag_0 = True
                x = x.unsqueeze(0)
            elif x.shape[1] == 1:
                flag_1 = True
                x = x.unsqueeze(1)
        out = self.gemm_forward_cuda(x)
        if flag_0:
            out = out.squeeze(0)
        if flag_1:
            out = out.squeeze(1)
        return out


@torch.no_grad()
def quantize_activation_per_token_absmax_full(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().clamp_(-q_max, q_max)
    return t, scales

class WALinear_online(nn.Module):
    def __init__(self, in_features, out_features, bias, dev, w_bit, a_bit, weight_group):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.weight_group = weight_group if weight_group != -1 else in_features
        
        assert self.in_features % self.weight_group == 0

        # W8 signed weight
        self.register_buffer('qweight', torch.zeros((out_features, in_features * w_bit // 8), dtype=torch.uint8, device=dev))

        # scales
        self.register_buffer('weight_scale', torch.zeros((out_features, in_features // self.weight_group), dtype=torch.float16, device=dev))
        self.register_buffer('weight_zero', torch.zeros((out_features, in_features // self.weight_group), dtype=torch.int8, device=dev))

        if bias:
            # fp16 bias
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear, weight_quant, act_quant, w_bit=4, a_bit=8, weight_group=128):
        qlinear = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, w_bit, a_bit, weight_group)
        
        # store bias
        if linear.bias is not None:
            qlinear.bias = linear.bias.clone().half()

        # store scales
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
    
    def gemm_forward_cuda(self, x_int8, act_scale_temp):
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

        x = x_int8.mul_(act_scale_temp)
        w = w_int8 * self.weight_scale.reshape(-1,1)
        w = w.reshape(w_int8_org_shape).to(torch.float16)

        out = F.linear(x, w, bias=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    @torch.no_grad()
    def forward(self, x):
        flag_0 = False
        flag_1 = False
        if x.dim() == 2:
            if x.shape[0] == 1:
                flag_0 = True
                x = x.unsqueeze(0)
            elif x.shape[1] == 1:
                flag_1 = True
                x = x.unsqueeze(1)
        x_int8, act_scale_temp = quantize_activation_per_token_absmax_full(x)
        out = self.gemm_forward_cuda(x_int8, act_scale_temp)
        if flag_0:
            out = out.squeeze(0)
        if flag_1:
            out = out.squeeze(1)
        return out