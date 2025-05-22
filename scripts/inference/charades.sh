output_dir=outputs/charades
pretrained_dir=aha_weights/
mkdir -vp  ${output_dir}/eval


# --------------------
# run inference
# --------------------
python -u -m test.inference --grounding_mode true \
    --test_dataset charades \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --lora_pretrained ${pretrained_dir} \
    --stream_end_prob_threshold 1 \
    --input_dir /data/charades/Charades_v1_480/ --frame_fps 2 --max_num_frames 400 \
    --test_fname datasets/charades/annotations/test-random_prompt.json \
    --score_heads "informative_score" \
    --output_fname ${output_dir}/eval/charades_test-random_prompt-pred.json \
    > ${output_dir}/eval/charades_test-random_prompt-pred.log 
wait

# --------------------
# grid search
# --------------------

python -u -m test.grid_search --test_dataset charades \
    --pred_file ${output_dir}/eval/charades_test-random_prompt-pred.json \
    --gold_file datasets/charades/annotations/test-random_prompt.json 
wait


# --------------------
# evaluate
# --------------------
python -u -m test.evaluate --func grounding \
    --pred_file ${output_dir}/eval/charades_test-random_prompt-pred.json \
    --gold_file datasets/charades/annotations/test-random_prompt.json \
    --output_file ${output_dir}/eval/charades_test-random_prompt-eval.json \
    > ${output_dir}/eval/charades_test-random_prompt-eval.log 
