# use mit-han-lab/pile-val-backup as the default calibration dataset
import os
# use local cached dataset
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetInfo
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default='llama2-7b-chat')
parser.add_argument('--tokenizer', default='/path/to/llama2-7b-chat')
parser.add_argument('--calib_dataset', default='mit-han-lab/pile-val-backup')
parser.add_argument('--output_path', default='../../data/calib_dataset')
parser.add_argument('--chunk_size', default=4096)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.pad_token = tokenizer.eos_token

info = DatasetInfo(description=args.model_id)

tgt_ds = {'text':[], 'token_ids': []}
ori_ds = load_dataset(args.calib_dataset, split='validation')

for i in tqdm(range(len(ori_ds))):
    item = ori_ds[i]['text']
    token_ids = tokenizer(item, add_special_tokens=True, max_length=args.chunk_size, truncation=True, return_tensors='pt')['input_ids'] # do not pad, only truncate to 4096 
    tgt_ds['text'].append(item)
    tgt_ds['token_ids'].append(token_ids)

tgt_ds = Dataset.from_dict(tgt_ds, info=info)
tgt_ds.save_to_disk(os.path.join(args.output_path, f'calib_dataset_{args.model_id}'))


    
    
    


