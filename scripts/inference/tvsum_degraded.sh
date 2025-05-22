
output_dir=outputs/tvsum_degraded
pretrained_dir=aha_weights/
mkdir -vp  ${output_dir}/eval

if [ -f "${output_dir}/eval/tvsum_test-random_prompt-pred.log" ]; then
    rm "${output_dir}/eval/tvsum_test-random_prompt-pred.log"
fi


# --------------------
# run inference
# --------------------
python -u -m test.inference --grounding_mode true \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --test_dataset tvsum_degraded \
    --skip_eval false \
    --caption_metadata_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv \
    --video_metadata_file datasets/tvsum/videos_metadata.json \
    --lora_pretrained ${pretrained_dir}  \
    --stream_end_prob_threshold 1 \
    --no_query False \
    --input_dir datasets/tvsum/ydata-tvsum50-v1_1/video --frame_fps 1 --max_num_frames 400 \
    --test_fname datasets/tvsum/annotations/test-random_prompt.json \
    --output_fname ${output_dir}/eval/tvsum_test-random_prompt-pred.json \
    > ${output_dir}/eval/tvsum_test-random_prompt-pred.log 
wait

# --------------------
# grid search
# --------------------
python -u -m test.grid_search --test_dataset tvsum_degraded \
    --pred_file ${output_dir}/eval/tvsum_test-random_prompt-pred.json \
    --gold_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv

# --------------------
# evaluate
# --------------------
python -u -m test.evaluate --func tvsum_degraded \
    --pred_file ${output_dir}/eval/tvsum_test-random_prompt-pred.json \
    --gold_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv \
    --output_file ${output_dir}/eval/tvsum_test-random_prompt-eval.json 