import torch
from kv_quant.quantization_offline.quant_funcs import pseudo_quantize_tensor
from kv_quant.quantization_offline.qlinear.sqwa import WALinear
from kv_quant.utils import get_module_by_name_suffix
from functools import partial


def quantize_model(model, args, quant_mix_gate=False, offline_cache=None, offline_excluded=None, return_param=False, max_len=4096):  
    """
    offline_cache: use offline-calculated scales and zeros from calib dataset to do quantization. pass the cache path
    return_param: return collected scales and zeros, used when generating scales and zeros on the offline stage  
    offline_excluded: a list of module names to be excluded from offline quant (use online scales and zeros). eg. ['down_proj']   
    """

    # KV cache quantization
    if args.kv_bit is not None and args.kv_bit > 0 and args.kv_bit < 16:
        model.config.kv_bit = args.kv_bit
        model.config.kv_group_size = args.kv_group_size
        
        # replace the attention module
        if 'chatglm3' in model.config._name_or_path.lower():
            from kv_quant.quantization.qattn.sw.glm3_attn import SelfAttention
            for i, block in enumerate(model.transformer.encoder.layers):
                new_attn = SelfAttention(model.config, block.self_attention.layer_number, block.self_attention.query_key_value.weight.device).half().to(block.self_attention.query_key_value.weight.device)
                new_attn.load_state_dict(block.self_attention.state_dict())
                block.self_attention = new_attn
        elif 'chatglm2' in model.config._name_or_path.lower():
            from kv_quant.quantization.qattn.sw.glm2_attn import SelfAttention
            for i, block in enumerate(model.transformer.encoder.layers):
                new_attn = SelfAttention(model.config, block.self_attention.layer_number, block.self_attention.query_key_value.weight.device).half().to(block.self_attention.query_key_value.weight.device)
                new_attn.load_state_dict(block.self_attention.state_dict())
                block.self_attention = new_attn
        elif 'opt' in model.config._name_or_path.lower():
            if model.config._flash_attn_2_enabled:
                print("Using flash attention for LLaMA model")
                from kv_quant.quantization.qattn.sw.opt_attn import OptFlashAttention2
                for i, block in enumerate(model.model.decoder.layers):
                    new_attn = OptFlashAttention2(model.config, block.self_attn.is_decoder).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
            else:
                from kv_quant.quantization.qattn.sw.opt_attn import OPTAttention
                for i, block in enumerate(model.model.decoder.layers):
                    new_attn = OPTAttention(model.config, block.self_attn.is_decoder).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
        elif 'bloom' in model.config._name_or_path.lower():
            from kv_quant.quantization.qattn.sw.bloom_attn import BloomAttention
            for i, block in enumerate(model.transformer.h):
                new_attn = BloomAttention(model.config).half().to(block.self_attention.query_key_value.weight.device)
                new_attn.load_state_dict(block.self_attention.state_dict())
                block.self_attention = new_attn
        elif 'llama' in model.config.architectures[0].lower() or 'vicuna' in model.config.architectures[0].lower() or 'longchat' in model.config.architectures[0].lower() or 'baichuan' in model.config.architectures[0].lower():
            if model.config._flash_attn_2_enabled:
                print("Using flash attention for LLaMA model")
                from kv_quant.quantization.qattn.sw.llama_attn import LlamaFlashAttention2
                for i, block in enumerate(model.model.layers):
                    new_attn = LlamaFlashAttention2(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
            else:
                from kv_quant.quantization.qattn.sw.llama_attn import LlamaAttention
                for i, block in enumerate(model.model.layers):
                    new_attn = LlamaAttention(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
        elif 'falcon' in model.config.architectures[0].lower():
            if model.config._attn_implementation == "flash_attention_2":
                print("Using flash attention for Falcon model")
                from kv_quant.quantization.qattn.sw.falcon_attn import FalconFlashAttention2
                for i, block in enumerate(model.transformer.h):
                    new_attn = FalconFlashAttention2(model.config).half().to(block.self_attention.query_key_value.weight.device)
                    new_attn.load_state_dict(block.self_attention.state_dict())
                    block.self_attention = new_attn
            else:
                from kv_quant.quantization.qattn.sw.falcon_attn import FalconAttention
                for i, block in enumerate(model.transformer.h):
                    new_attn = FalconAttention(model.config).half().to(block.self_attention.query_key_value.weight.device)
                    new_attn.load_state_dict(block.self_attention.state_dict())
                    block.self_attention = new_attn
        elif 'stable' in model.config.architectures[0].lower():
            if model.config._flash_attn_2_enabled:
                print("Using flash attention for LLaMA model")
                from kv_quant.quantization.qattn.sw.stable_attn import FlashAttention2
                for i, block in enumerate(model.model.layers):
                    new_attn = FlashAttention2(model.config).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
            else:
                from kv_quant.quantization.qattn.sw.stable_attn import Attention
                for i, block in enumerate(model.model.layers):
                    new_attn = Attention(model.config).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
        elif 'mistral' in model.config.architectures[0].lower():
            if model.config._attn_implementation == "flash_attention_2":
                print("Using flash attention for Mistral model")
                from kv_quant.quantization.qattn.sw.mistral_attn import MistralFlashAttention2
                for i, block in enumerate(model.model.layers):
                    new_attn = MistralFlashAttention2(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
            else:
                from kv_quant.quantization.qattn.sw.mistral_attn import MistralAttention
                for i, block in enumerate(model.model.layers):
                    new_attn = MistralAttention(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
        elif 'mixtral' in model.config.architectures[0].lower():
            if model.config._attn_implementation == "flash_attention_2":
                print("Using flash attention for Mixtral model")
                from kv_quant.quantization.qattn.sw.mixtral_attn import MixtralFlashAttention2
                for i, block in enumerate(model.model.layers):
                    new_attn = MixtralFlashAttention2(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
            else:
                from kv_quant.quantization.qattn.sw.mixtral_attn import MixtralAttention
                for i, block in enumerate(model.model.layers):
                    new_attn = MixtralAttention(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
        elif 'mpt' in model.config.architectures[0].lower():
            # from kv_quant.quantization.qattn.sw.mpt_attn import MptAttention
            # for i, block in enumerate(model.transformer.blocks):
            #     new_attn = MptAttention(model.config).half().to(block.attn.Wqkv_proj.weight.device)
            #     new_attn.load_state_dict(block.attn.state_dict())
            #     block.attn = new_attn
            from kv_quant.quantization.qattn.sw.mpt_attn import flash_attn_fn, scaled_multihead_dot_product_attention, triton_flash_attn_fn
            for i, block in enumerate(model.transformer.blocks):
                if model.config.attn_config['attn_impl'] == 'flash':
                    block.attn.attn_fn = partial(flash_attn_fn, kv_bit=model.config.kv_bit, kv_group_size=model.config.kv_group_size)
                elif model.config.attn_config['attn_impl'] == 'triton':
                    block.attn.attn_fn = partial(triton_flash_attn_fn, kv_bit=model.config.kv_bit, kv_group_size=model.config.kv_group_size)
                elif model.config.attn_config['attn_impl'] == 'torch':
                    block.attn.attn_fn = partial(scaled_multihead_dot_product_attention, kv_bit=model.config.kv_bit, kv_group_size=model.config.kv_group_size)
                else:
                    raise NotImplementedError(f"Not support attention implementation: {model.config.attn_config['attn_impl']}")
        elif 'gemma' in model.config.architectures[0].lower():
            if model.config._attn_implementation == "flash_attention_2":
                print("Using flash attention for Gemma model")
                from kv_quant.quantization.qattn.sw.gemma_attn import GemmaFlashAttention2
                for i, block in enumerate(model.model.layers):
                    new_attn = GemmaFlashAttention2(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
            else:
                from kv_quant.quantization.qattn.sw.gemma_attn import GemmaAttention
                for i, block in enumerate(model.model.layers):
                    new_attn = GemmaAttention(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                    new_attn.load_state_dict(block.self_attn.state_dict())
                    block.self_attn = new_attn
        else:
            raise NotImplementedError(f"Not support model architecture: {model.config.architectures[0]}")
        torch.cuda.empty_cache()
    else:
        if 'mpt' in model.config.architectures[0].lower():
            from kv_quant.quantization.qattn.sw.mpt_attn import flash_attn_fn, scaled_multihead_dot_product_attention, triton_flash_attn_fn
            for i, block in enumerate(model.transformer.blocks):
                if model.config.attn_config['attn_impl'] == 'flash':
                    block.attn.attn_fn = partial(flash_attn_fn, kv_bit=model.config.kv_bit, kv_group_size=model.config.kv_group_size)
                elif model.config.attn_config['attn_impl'] == 'triton':
                    block.attn.attn_fn = partial(triton_flash_attn_fn, kv_bit=model.config.kv_bit, kv_group_size=model.config.kv_group_size)
                elif model.config.attn_config['attn_impl'] == 'torch':
                    block.attn.attn_fn = partial(scaled_multihead_dot_product_attention, kv_bit=model.config.kv_bit, kv_group_size=model.config.kv_group_size)
        torch.cuda.empty_cache()

    # Weight-only quantization
    if (args.w_bit is not None and args.w_bit < 16) and (args.a_bit is None or args.a_bit >= 16):
        assert args.w_bit > 0 and args.w_bit < 16, "Weight bitwidth should be an integer between [1, 16] for weigth-only quantization, please check."
        # Use original Linear module
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                if not quant_mix_gate and 'gate' in name and 'mixtral' in model.config.architectures[0].lower():
                    pass
                try:
                    module.weight.data = pseudo_quantize_tensor(module.weight.data, n_bits=args.w_bit, q_group_size=args.w_group_size)
                    torch.cuda.empty_cache()
                except:
                    print(f"Failed to quantize {name}")

    # Weight-Activation quantization (offline + online)
    if args.w_bit is not None and args.w_bit > 0 and args.w_bit < 16 and args.a_bit is not None and args.a_bit > 0 and args.a_bit < 16:
        # Replace original Linear module
        if offline_cache:
            import pickle
            with open(offline_cache, 'rb') as f:
                offline_cache = pickle.load(f)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                # determine offline_cache
                if not offline_cache:   # pure online quantization
                    offline_cache_this = None
                else:                   # offline quantization, use pre-generated scales and zeros
                    offline_cache_this = offline_cache[name]
                    if offline_excluded:    
                        for patt in offline_excluded:
                            if patt.lower() in name.lower():    # some modules use online quantization, while others use offline scales and zeros
                                offline_cache_this = None   

                new_linear = WALinear.from_float(module, weight_quant='per_group', act_quant=args.act_quant, zero_point=args.zero_point, w_bit=args.w_bit, a_bit=args.a_bit, \
                    weight_group=args.w_group_size, quantize_output=False, offline_cache=offline_cache_this, return_param=return_param, max_len=max_len)
                father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                setattr(father_module, name.split('.')[-1], new_linear)
                del new_linear, module
                torch.cuda.empty_cache()
                # eval("{}.{} = new_linear".format(father_module, name.split('.')[-1]))
    return model





def quantize_model_hw(model, args):
    # KV cache quantization
    if args.kv_bit is not None and args.kv_bit > 0 and args.kv_bit < 16:
        model.config.kv_bit = args.kv_bit
        model.config.kv_group_size = args.kv_group_size

        # replace the attention module
        if 'chatglm' in model.config.architectures[0].lower():
            from kv_quant.quantization.qattn.hw.glm3_attn import SelfAttention
            for i, block in enumerate(model.transformer.encoder.layers):
                new_attn = SelfAttention(model.config, block.self_attention.layer_number, block.self_attention.query_key_value.weight.device).half().to(block.self_attention.query_key_value.weight.device)
                new_attn.load_state_dict(block.self_attention.state_dict())
                block.self_attention = new_attn
        elif 'llama' in model.config.architectures[0].lower() or 'vicuna' in model.config.architectures[0].lower() or 'longchat' in model.config.architectures[0].lower() or 'baichuan' in model.config.architectures[0].lower() or 'chat' in model.config.architectures[0].lower():
            from kv_quant.quantization.qattn.hw.llama_attn import LlamaAttention
            for i, block in enumerate(model.model.layers):
                new_attn = LlamaAttention(block.self_attn.config).half().to(block.self_attn.q_proj.weight.device)
                new_attn.load_state_dict(block.self_attn.state_dict())
                block.self_attn = new_attn
        elif 'tele' in model.config.architectures[0].lower():
            from kv_quant.quantization.qattn.hw.tele_attn import TelechatAttention
            for i, block in enumerate(model.transformer.h):
                new_attn = TelechatAttention(block.self_attention.config, block.self_attention.layer_idx).half().to(block.self_attention.query.weight.device)
                new_attn.load_state_dict(block.self_attention.state_dict())
                block.self_attention = new_attn
        else:
            raise NotImplementedError(f"Not support model architecture: {model.config.architectures[0]}")

    # Weight-only quantization
    if args.w_bit is not None and args.w_bit > 0 and args.w_bit < 16 and args.a_bit >= 16:
        from kv_quant.quantization.qlinear.hqw import WLinear
        from kv_quant.utils import get_module_by_name_suffix
        # Replace original Linear module
        # Use original Linear module
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                new_linear = WLinear.from_float(module, w_bit=args.w_bit, weight_group=args.w_group_size)
                father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                setattr(father_module, name.split('.')[-1], new_linear)

    # Weight-Activation quantization
    if args.w_bit is not None and args.w_bit > 0 and args.w_bit < 16 and args.a_bit is not None and args.a_bit > 0 and args.a_bit < 16:
        from kv_quant.quantization.qlinear.hqwa import WALinear_online
        from kv_quant.utils import get_module_by_name_suffix
        # Replace original Linear module
        # Use original Linear module
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                new_linear = WALinear_online.from_float(module, weight_quant='per_group', act_quant='per_token', w_bit=args.w_bit, a_bit=args.a_bit, weight_group=args.w_group_size)
                father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                setattr(father_module, name.split('.')[-1], new_linear)
    return model

def quantize_model_hw_init(model, args):
    # KV cache quantization
    if args.kv_bit is not None and args.kv_bit > 0 and args.kv_bit < 16:
        model.config.kv_bit = args.kv_bit
        model.config.kv_group_size = args.kv_group_size

        # replace the attention module
        if 'chatglm' in model.config.architectures[0].lower():
            from kv_quant.quantization.qattn.hw.glm3_attn import SelfAttention
            for i, block in enumerate(model.transformer.encoder.layers):
                new_attn = SelfAttention(model.config, block.self_attention.layer_number, block.self_attention.query_key_value.weight.device).half().to(block.self_attention.query_key_value.weight.device)
                block.self_attention = new_attn
        elif 'llama' in model.config.architectures[0].lower() or 'vicuna' in model.config.architectures[0].lower() or 'longchat' in model.config.architectures[0].lower() or 'baichuan' in model.config.architectures[0].lower() or 'chat' in model.config.architectures[0].lower():
            from kv_quant.quantization.qattn.hw.llama_attn import LlamaAttention
            for i, block in enumerate(model.model.layers):
                new_attn = LlamaAttention(block.self_attn.config).half().to(block.self_attn.q_proj.weight.device)
                block.self_attn = new_attn
        elif 'tele' in model.config.architectures[0].lower():
            from kv_quant.quantization.qattn.hw.tele_attn import TelechatAttention
            for i, block in enumerate(model.transformer.h):
                new_attn = TelechatAttention(block.self_attention.config, block.self_attention.layer_idx).half().to(block.self_attention.query.weight.device)
                block.self_attention = new_attn
        else:
            raise NotImplementedError(f"Not support model architecture: {model.config.architectures[0]}")

    # Weight-only quantization
    if args.w_bit is not None and args.w_bit > 0 and args.w_bit < 16 and args.a_bit >= 16:
        from kv_quant.quantization.qlinear.hqw import WLinear
        from kv_quant.utils import get_module_by_name_suffix
        # Replace original Linear module
        # Use original Linear module
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                new_linear = WLinear.from_float(module, w_bit=args.w_bit, weight_group=args.w_group_size, init_only=True)
                father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                setattr(father_module, name.split('.')[-1], new_linear)

    # Weight-Activation quantization
    if args.w_bit is not None and args.w_bit > 0 and args.w_bit < 16 and args.a_bit is not None and args.a_bit > 0 and args.a_bit < 16:
        from kv_quant.quantization.qlinear.hqwa import WALinear_online
        from kv_quant.utils import get_module_by_name_suffix
        # Replace original Linear module
        # Use original Linear module
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                new_linear = WALinear_online.from_float(module, weight_quant='per_group', act_quant='per_token', w_bit=args.w_bit, a_bit=args.a_bit, weight_group=args.w_group_size, init_only=True)
                father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                setattr(father_module, name.split('.')[-1], new_linear)
    return model
