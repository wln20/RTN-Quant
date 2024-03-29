import argparse
import torch

from kv_quant.quantization.quant_wrapper import quantize_model
from kv_quant.utils import build_model_and_enc

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--model_id", type=str, help="model_id")
parser.add_argument("--output_path", type=str, default=None, help="path to save the quantized model")
parser.add_argument("--use_flash_attn", action="store_true")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)
args = parser.parse_args()


def main():
    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)
    
    # quantize model
    print(f'* Quantizing: W{args.w_bit} A{args.a_bit} KV{args.kv_bit}')
    model = quantize_model(model, args)

    # save the quantized model
    if args.output_path:
        model.save_pretrained(args.output_path, safe_serialization=False)
        enc.save_pretrained(args.output_path)

    # evaluation
    print('* Generating ...')
    print('='*40)
    prompt = "Tell me what is 'Gettysburg Address' about: "
    print(f'Prompt: {prompt}')
    print('-'*40)
    input_ids = enc(prompt, return_tensors="pt")['input_ids'].to(next(model.parameters()).device)
    output = model.generate(input_ids, do_sample=True, max_length=200, top_p=0.95, top_k=60)
    print(f'Output: {enc.decode(output[0])}')
    print('='*40)


if __name__ == "__main__":
    main()