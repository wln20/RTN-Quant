import torch
from functools import partial

def prune_model_kv(model, args):
    if 'llama' in model.config.architectures[0].lower() or 'vicuna' in model.config.architectures[0].lower() or 'longchat' in model.config.architectures[0].lower() or 'baichuan' in model.config.architectures[0].lower():
            from kv_quant.kv_pruning.attn.llama_attn import LlamaAttention
            for i, block in enumerate(model.model.layers):
                new_attn = LlamaAttention(model.config, block.self_attn.layer_idx).half().to(block.self_attn.q_proj.weight.device)
                new_attn.load_state_dict(block.self_attn.state_dict())
                block.self_attn = new_attn
    else:
        raise NotImplementedError(f"Not support model architecture: {model.config.architectures[0]}")

    torch.cuda.empty_cache()
    return model



