output_dir=/mnt/training-data/tvsum_trained_output
pretrained_dir=outputs/mmduet
mkdir -vp $output_dir

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_VISIBLE_DEVICES=0,1

PYTHONWARNINGS="ignore" torchrun --nproc_per_node 2 --master_port 29506 train.py --deepspeed configs/deepspeed/zero2offload.json \
    --bf16 false --fp16 true \
    --dataset_config configs/datasets/ahait.json \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov \
    --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --evaluation_strategy no --prediction_loss_only false \
    --save_strategy steps --save_steps 500 --save_total_limit 5 \
    --learning_rate 0.00002 --optim adamw_torch --lr_scheduler_type cosine --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --logging_steps 10 \
    --report_to wandb \
    --output_dir $output_dir \
    > $output_dir/train.log 2>&1 &
