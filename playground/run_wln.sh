CUDA_VISIBLE_DEVICES='0' python basic_quant_kv.py --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat
CUDA_VISIBLE_DEVICES='0' python basic_quant_kv.py --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat_quant_w_8_a_8 --w_bit 8 --a_bit 8
CUDA_VISIBLE_DEVICES='0' python basic_quant_kv.py --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat_quant_w_4_a_8 --w_bit 4 --a_bit 8
CUDA_VISIBLE_DEVICES='0' python basic_quant_kv.py --model_path /share/datasets/public_models/Llama-2-7b-chat-hf --model_id llama2-7b-chat_quant_w_4_a_4 --w_bit 4 --a_bit 4
