# Source: https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/training/run_pt.sh
lr=2e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model="meta-llama/Llama-2-7b-hf"
chinese_tokenizer_path="../../../checkpoints/merged_tokenizer_hf"
dataset_dir="../../../data/CulturaX/text/"
data_cache=temp_data_cache_dir
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=2
output_dir=output_dir

deepspeed_config_file=ds_zero2_no_offload.json

python run_clm_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --do_train \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 50 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --gradient_checkpointing \
    --fp16
