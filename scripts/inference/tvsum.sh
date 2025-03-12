
output_dir=outputs/tvsum_eval
pretrained_dir=outputs/mmduet
mkdir -vp  ${output_dir}/eval


# --------------------
# run inference
# --------------------
python -u -m test.inference --grounding_mode true \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --test_dataset tvsum \
    --skip_eval true \
    --caption_metadata_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv \
    --video_metadata_file datasets/tvsum/videos_metadata.json \
    --lora_pretrained ${pretrained_dir} \
    --stream_end_prob_threshold 1 \
    --input_dir datasets/tvsum/ydata-tvsum50-v1_1/video --frame_fps 2 --max_num_frames 400 \
    --test_fname datasets/tvsum/annotations/test-random_prompt.json \
    --output_fname ${output_dir}/eval/tvsum_test-random_prompt-pred.json \
    > ${output_dir}/eval/tvsum_test-random_prompt-pred.log 2>&1 &
wait


# --------------------
# evaluate
# --------------------
python -u -m test.evaluate --func tvsum \
    --pred_file ${output_dir}/eval/tvsum_test-random_prompt-pred.json \
    --gold_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv \
    --output_file ${output_dir}/eval/tvsum_test-random_prompt-eval.json \
    > ${output_dir}/eval/tvsum_test-random_prompt-eval.log 2>&1 &