import argparse
import os
import torch
from tqdm import tqdm
import pickle
from datasets import load_from_disk

from kv_quant.quantization_offline.quant_wrapper import quantize_model
from kv_quant.quantization_offline.qlinear.sqwa import WALinear
from kv_quant.utils import build_model_and_enc

parser = argparse.ArgumentParser()
# model settings
parser.add_argument("--model_path", type=str, default='path/to/llama2-7b-chat', help="path of the hf model")
parser.add_argument("--model_id", type=str, default='llama2-7b-chat', help="model_id")
parser.add_argument("--output_path", type=str, default=None, help="path to save the quantized model")
# calibration quant settings
parser.add_argument("--act_quant", type=str, default='per_token', choices=['per_token', 'per_tensor'])
parser.add_argument("--zero_point", type=bool, default=True) 
parser.add_argument("--calib_dataset_root", type=str, default='../../data/calib_dataset')
parser.add_argument("--calib_proportion_div", type=float, default=5, help='use 1/n of the calib dataset to calib')
parser.add_argument("--cache_save_root", default='../../data/quant_cache')
# quant settings
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=16)
# not implemented
parser.add_argument("--use_flash_attn", action="store_true")
parser.add_argument("--kv_group_size", type=int, default=128)
parser.add_argument("--kv_bit", type=int, default=16)
args = parser.parse_args()


def main():
    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path, args.use_flash_attn, args.kv_bit, args.kv_group_size)

    # quantize model
    print(f'* Quantizing: W{args.w_bit} A{args.a_bit}')
    model = quantize_model(model, args, return_param=True, max_len=model.config.max_position_embeddings)    # only support llama now

    # save the quantized model
    if args.output_path:
        model.save_pretrained(args.output_path, safe_serialization=False)
        enc.save_pretrained(args.output_path)

    # generate scales
    print('* Generating scales and zeros ...')
    calib_dataset_path = os.path.join(args.calib_dataset_root, f'calib_dataset_{args.model_id}')
    calib_dataset = load_from_disk(calib_dataset_path)
    assert args.model_id == calib_dataset.info.description, f"You are using {args.model_id} to generate quant cache, \
        but the calib dataset is tokenized by {calib_dataset.info.description}'s tokenizer which is incompatible!"

    for i in tqdm(range(len(calib_dataset)//args.calib_proportion_div)):    # must let batch_size = 1
        try:
            input_ids = torch.tensor(calib_dataset[i]['token_ids']).to(model.device)
            output = model.generate(input_ids, use_cache=False, max_new_tokens=1)
        except:
            continue
        # out = tokenizer.decode(output[0])
    
    # create the dictionary to store scales and zeros
    cache_dict = {}
    # calculate mean scales and zeros
    for name, module in model.named_modules():
        if isinstance(module, WALinear) and 'lm_head' not in name and 'output_layer' not in name:
            cache_dict[name] = (module.act_scales/module.num_acts, module.act_zeros/module.num_acts)  # ([4096, 1], [4096, 1])
    
    # save results
    print('* Saving scales and zeros ...')
    cache_save_path = os.path.join(args.cache_save_root, args.model_id)
    os.makedirs(cache_save_path, exist_ok=True)
    with open(os.path.join(cache_save_path, f"a{args.a_bit}_{args.act_quant}_{'with_zero_point' if args.zero_point else 'without_zero_point'}.pkl"), 'wb') as f:
        pickle.dump(cache_dict, f)

if __name__ == "__main__":
    main()