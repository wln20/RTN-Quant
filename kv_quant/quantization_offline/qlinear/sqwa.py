import torch
from torch import nn
from functools import partial
from kv_quant.quantization_offline.quant_funcs import *

class WALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', a_bit=8, w_bit=8, quantize_output=False, dev='cuda', offline_cache=None, return_param=False, max_len=4096):
        """
        offline_cache: use offline-generated scales and zeros for quantization
        return_param: whether to return calculated scales and zeros, used at the offline-generation stage on the calib dataset
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.return_param = return_param
        if self.return_param:
            # average scales and zeros on calib dataset, only support per_token act quant now
            # in this module, the act_scales and act_zeros are the sum of all the activation features. 
            # need to do average when calculating
            self.act_scales = torch.zeros(max_len, 1)
            self.act_zeros = torch.zeros(max_len, 1)
            self.num_acts = 0
        # the position_id of the current token (only applicable to decoding stage)   
        self.position_id = 0

        self.register_buffer('weight', torch.zeros(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=self.a_bit, offline_cache=offline_cache, return_param=return_param)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=self.a_bit, offline_cache=offline_cache, return_param=return_param)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WALinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        # x: [1, 13, 4096]. when using batch, it's like: [2, 13, 4096]. when decoding, it's like: [1, 1, 4096]

        # return scales and activations, used when generating scales and activations offline
        if self.return_param:   
            q_x, scales, zeros = self.act_quant(x, position_id=self.position_id)
            self.act_scales[: scales.shape[0]] += scales.cpu()
            self.act_zeros[: zeros.shape[0]] += zeros.cpu()
            self.num_acts += 1
        else:  
            q_x = self.act_quant(x, position_id=self.position_id)

        # update position
        # eg. At prefill stage assume the sequence is: [token_0, ..., token_8], seq_len=9, set self.position_id=9.
        # Then token_9 comes, its position_id should be 9, so first pass the original self.position_id to self.act_quant, then update self.postion_id to be 9+1=10
        seq_len = x.shape[1] if x.dim() == 3 else x.shape[0]
        if seq_len > 1: # prefill stage
            self.position_id = seq_len
        else:           # decode stage
            self.position_id += 1

        y = torch.functional.F.linear(q_x.to(self.weight.dtype), self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', w_bit=4, a_bit=8, weight_group=128, quantize_output=False, offline_cache=None, return_param=False, max_len=4096):
        assert isinstance(module, torch.nn.Linear)
        new_module = WALinear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, a_bit=a_bit, w_bit=w_bit, quantize_output=quantize_output, dev=module.weight.device, offline_cache=offline_cache, return_param=return_param, max_len=max_len)
        
        # Quantize the weight matrices
        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(module.weight, n_bits=w_bit)
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(module.weight, n_bits=w_bit)
        elif weight_quant == 'per_group':
            new_module.weight = pseudo_quantize_tensor(module.weight, n_bits=w_bit, q_group_size=weight_group, inplace=True)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        del module
        return new_module

    def __repr__(self):
        return 'W{}A{}Linear'.format(self.w_bit, self.a_bit)

