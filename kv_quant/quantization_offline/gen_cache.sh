CUDA_VISIBLE_DEVICES='4,5,6' python data_prepare.py --model_id llama2-7b --tokenizer /share/datasets/public_models/Llama-2-7b-hf
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 8 --model_path /share/datasets/public_models/Llama-2-7b-hf --model_id llama2-7b
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 4 --model_path /share/datasets/public_models/Llama-2-7b-hf --model_id llama2-7b
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 3 --model_path /share/datasets/public_models/Llama-2-7b-hf --model_id llama2-7b
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 2 --model_path /share/datasets/public_models/Llama-2-7b-hf --model_id llama2-7b

CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 8 --model_path /share/datasets/public_models/Llama-2-7b-hf --model_id llama2-7b --act_quant per_tensor
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 4 --model_path /share/datasets/public_models/Llama-2-7b-hf --model_id llama2-7b --act_quant per_tensor
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 3 --model_path /share/datasets/public_models/Llama-2-7b-hf --model_id llama2-7b --act_quant per_tensor
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 2 --model_path /share/datasets/public_models/Llama-2-7b-hf --model_id llama2-7b --act_quant per_tensor

CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 8 --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat 
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 4 --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 3 --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 2 --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat

CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 8 --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat --act_quant per_tensor
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 4 --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat --act_quant per_tensor
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 3 --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat --act_quant per_tensor
CUDA_VISIBLE_DEVICES='4,5,6' python quant_cache_generator.py --w_bit 8 --a_bit 2 --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat --act_quant per_tensor