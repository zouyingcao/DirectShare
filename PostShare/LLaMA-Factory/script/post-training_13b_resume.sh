cd ..

train_name="train_llama2-13b-cut30_en_wiki_1_resume"
nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=9912 src/train_bash.py \
    --deepspeed ds_config_zero1.json \
    --resume_from_checkpoint ../output/train_llama2-13b-cut30_en_wiki_1/checkpoint-92 \
    --ignore_data_skip True \
    --stage pt \
    --do_train \
    --model_name_or_path 'llama-ckpt-hf/Llama-2-13b-hf/' \
    --dataset en_wiki \
    --cache_path ../dataset_cache/en_wiki\
    --finetuning_type full \
    --template default \
    --output_dir ../output/$train_name \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --report_to tensorboard \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --save_steps 0.01 \
    --learning_rate 5e-5 \
    --num_train_epochs 0.01 \
    --plot_loss \
    --tf32 True \
    --bf16 True >> ../logs/$train_name.log 2>&1 &
    