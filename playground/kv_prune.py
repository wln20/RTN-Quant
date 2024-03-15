import argparse
import os
os.environ['CUDA_VISILE_DEVICES'] = '7'

from kv_quant.kv_pruning.kv_prune_wrapper import prune_model_kv
from kv_quant.utils import build_model_and_enc

parser = argparse.ArgumentParser()
# model settings
parser.add_argument("--model_path", type=str, default='/share/datasets/public_models/Llama-2-7b-chat-hf', help="path of the hf model")
parser.add_argument("--model_id", type=str, default='llama2-7b-chat', help="model_id")
args = parser.parse_args()

def main():
    model, enc = build_model_and_enc(args.model_path)
    enc.pad_token = enc.eos_token

    # simple evaluation
    print('* Generating ...')
    print('='*40)
    prompts = ["Give a brief introduction of China: ", "Introduce Japan: "]
    print(f'Prompts: {prompts}')
    print('-'*40)
    input_ids = enc(prompts, padding=True, return_tensors="pt")['input_ids'].to(next(model.parameters()).device)
    output = model.generate(input_ids, use_cache=True, do_sample=True, max_new_tokens=200, top_p=0.95, top_k=60)
    for i, prompt in enumerate(prompts):
        print(f'Output_{i}: {enc.decode(output[i])}')
        print('-'*20)
    print('='*40)


    print(f"* Setting kv_prune attention layers for model {args.model_id}")
    model = prune_model_kv(model, args)

    # simple evaluation
    print('* Generating ...')
    print('='*40)
    prompts = ["Give a brief introduction of China: ", "Introduce Japan: "]
    print(f'Prompts: {prompts}')
    print('-'*40)
    input_ids = enc(prompts, padding=True, return_tensors="pt")['input_ids'].to(next(model.parameters()).device)
    output = model.generate(input_ids, use_cache=True, do_sample=True, max_new_tokens=200, top_p=0.95, top_k=60)
    for i, prompt in enumerate(prompts):
        print(f'Output_{i}: {enc.decode(output[i])}')
        print('-'*20)
    print('='*40)


if __name__ == '__main__':
    main()