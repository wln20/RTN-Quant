import argparse
import os
import torch
from tqdm import tqdm
import pickle
from datasets import load_from_disk

from kv_quant.quantization_offline.quant_wrapper import quantize_model
from kv_quant.utils import build_model_and_enc
from kv_quant.evaluation.lm_evaluator import evaluate

parser = argparse.ArgumentParser()
# model settings
parser.add_argument("--model_path", type=str, default='/share/datasets/public_models/Llama-2-7b-chat-hf', help="path of the hf model")
parser.add_argument("--model_id", type=str, default='llama2-7b-chat', help="model_id")
parser.add_argument("--output_path", type=str, default=None, help="path to save the quantized model")
# offline quant settings
parser.add_argument("--offline_cache_path", type=str, default='../data/quant_cache/llama2-7b-chat/a8.pkl')
parser.add_argument("--offline_excluded", type=str, default=None)   # separate with comma, eg. "down_proj,up_proj"
# evaluation settings
parser.add_argument("--eval_tasks", type=str, default='wikitext')   # separate with comma, eg. "wikitext,lambada,hellaswag"
parser.add_argument("--eval_limit", type=int, default=None)
parser.add_argument("--eval_batch_size", type=int, default=1)
# quantization bitwidth settings
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=4)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=8)
# TODO: not supported, just keep the interface compatible with the original script
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)
parser.add_argument("--use_flash_attn", action="store_true")
args = parser.parse_args()


def main():
    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
    enc.pad_token = enc.eos_token

    # full evaluation on original model
    if args.eval_tasks:
        evaluate(args.model_id, model, enc, args.eval_tasks, quant_bitwidth=None, limit=args.eval_limit, batch_size=args.eval_batch_size, max_length=4096)

    # quantize model
    if args.offline_excluded:
        offline_excluded = args.offline_excluded.split(',')
    else:
        offline_excluded = None
    print(f'* Quantizing: W{args.w_bit} A{args.a_bit}')
    print(f'* Offline-excluded: {offline_excluded}')

    model = quantize_model(model, args, offline_cache=args.offline_cache_path, offline_excluded=offline_excluded)    # only support llama now

    # save the quantized model
    if args.output_path:
        model.save_pretrained(args.output_path, safe_serialization=False)
        enc.save_pretrained(args.output_path)

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

    # full evaluation on quant model
    if args.eval_tasks:
        evaluate(args.model_id, model, enc, args.eval_tasks, quant_bitwidth=f'W{args.w_bit} A{args.a_bit}', limit=args.eval_limit, batch_size=args.eval_batch_size, max_length=4096)

if __name__ == "__main__":
    main()