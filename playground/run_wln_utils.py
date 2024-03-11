import sys
import os

path_dict = {
    # 'opt-1.3b': '/share/datasets/public_models/facebook_opt-1.3b',
    # 'opt-2.7b': '/share/datasets/public_models/facebook_opt-2.7b',
    # 'opt-6.7b': '/share/datasets/public_models/facebook_opt-6.7b',
    # 'opt-13b': '/share/datasets/public_models/facebook_opt-13b',
    # 'opt-30b': '/share/datasets/public_models/facebook_opt-30b',
    # 'opt-66b': '/share/datasets/public_models/facebook_opt-66b',
    # 'vicuna-7b-v1.5': '/share/datasets/public_models/lmsys_vicuna-7b-v1.5',
    # 'vicuna-7b-v1.5-16k': '/share/datasets/public_models/lmsys_vicuna-7b-v1.5-16k',
    # 'vicuna-13b-v1.5': '/share/datasets/public_models/lmsys_vicuna-13b-v1.5',
    # 'vicuna-13b-v1.5-16k': '/share/datasets/public_models/lmsys_vicuna-13b-v1.5-16k',
    'llama2-7b-chat': '/share/datasets/public_models/Llama-2-7b-chat-hf',
    # 'llama2-13b-chat': '/share/datasets/public_models/Llama-2-13b-chat-hf',
    # 'llama2-70b-chat': '/share/datasets/public_models/Llama-2-70b-chat-hf',
    # 'chatglm3-6b': '/share/datasets/public_models/THUDM_chatglm3-6b',
    # 'chatglm3-6b-32k': '/share/datasets/public_models/THUDM_chatglm3-6b-32k',
    # 'longchat-7b-16k': '/share/datasets/public_models/lmsys_longchat-7b-16k',
    # 'longchat-7b-v1.5-32k': '/share/datasets/public_models/lmsys_longchat-7b-v1.5-32k',
    # 'longchat-13b-16k': '/share/datasets/public_models/lmsys_longchat-13b-16k',
    # 'falcon-7b-instruct': '/share/datasets/public_models/tiiuae_falcon-7b-instruct',
    # 'falcon-40b-instruct': '/share/datasets/public_models/tiiuae_falcon-40b-instruct',
    # 'falcon-180b-chat': '/share/datasets/public_models/tiiuae_falcon-180B-chat',
    # 'mixtral-8x7b': '/share/datasets/public_models/mistralai_Mixtral-8x7B-Instruct-v0.1/',
    # 'mistral-7b': '/share/datasets/tmp_share/lsy/self_models/Mistral-7B-Instruct-v0.2',
    # 'gemma-2b-it': '/share/datasets/public_models/gemma-2b-it',
    # 'stablelm-zephyr-7b': '/share/wangluning/Huggingface/hub/models--stabilityai--stablelm-zephyr-3b/snapshots/8b471c751c0e78cb46cf9f47738dd0eb45392071',
    # 'gemma-7b-it': '/share/datasets/public_models/gemma-7b-it',
    # 'mamba-2b8-chat': '/share/wangluning/Huggingface/hub/models--havenhq--mamba-chat/snapshots/d343f8ade4c870d916b362746dd23821aae132dd'
    # 'phi-2': '/share/wangluning/Huggingface/hub/models--microsoft--phi-2/snapshots/b10c3eba545ad279e7208ee3a5d644566f001670'
    # 'bloom-560m': '/share/datasets/public_models/bigscience_bloom-560m',
    # 'bloom-1b1': '/share/datasets/public_models/bigscience_bloom-1b1',
    # 'bloom-1b7': '/share/datasets/public_models/bigscience_bloom-1b7',
    # 'bloom-3b': '/share/datasets/public_models/bigscience_bloom-3b',
    # 'bloom-7b1': '/share/datasets/public_models/bigscience_bloom-7b1',
}

### modify here
models = list(path_dict.keys())
mode = 'wa'

gpu = '0'
run_file = 'run_wln.sh'


bit_widths = {
    'w': [4,3,2],
    'wa': [(8,8), (4,8), (4,4)],
    'kv': [4,3,2],
    'wkv': [(4,4)]
}

          
with open(run_file, 'w') as f:
    # step1: gen_model_answer
    for model in models:
        # baseline
        f.write(f"CUDA_VISIBLE_DEVICES='{gpu}' python basic_quant_kv.py --model_path {path_dict[model]} --model_id {model}\n")
        #model_ids.append(model)
        # quant 
        if mode == 'kv' or mode == 'w':
            for wbit in bit_widths[mode]:
                f.write(f"CUDA_VISIBLE_DEVICES='{gpu}' python basic_quant_kv.py --model_path {path_dict[model]} --model_id {model}_quant_{mode}_{wbit} --{mode}_bit {wbit} {f'--{mode}_group_size 64' if 'falcon' in model else f'--{mode}_group_size 128'}\n")
        elif mode == 'wa':
            for wbit, abit in bit_widths[mode]:
                f.write(f"CUDA_VISIBLE_DEVICES='{gpu}' python basic_quant_kv.py --model_path {path_dict[model]} --model_id {model}_quant_w_{wbit}_a_{abit} --w_bit {wbit} --a_bit {abit}\n" )
        elif mode == 'wkv':   
            for wbit, kvbit in bit_widths[mode]:
                f.write(f"CUDA_VISIBLE_DEVICES='{gpu}' python basic_quant_kv.py --model_path {path_dict[model]} --model_id {model}_quant_w_{wbit}_kv_{kvbit} --w_bit {wbit} --kv_bit {kvbit} {f'--w_group_size 64 --kv_group_size 64' if 'falcon' in model else f'--w_group_size 128 --kv_group_size 128'}\n")

