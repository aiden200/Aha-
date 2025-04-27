output_dir=outputs/arl_scout
pretrained_dir=aiden200/aha
mkdir -vp  ${output_dir}/eval
thres_sum=2

# # --------------------
# # run inference
# # --------------------
python -u -m test.inference \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --test_dataset arl_scout \
    --skip_eval false \
    --anno_file test/arl_scout/p1.02_main1/all-images \
    --lora_pretrained ${pretrained_dir}  \
    --stream_end_score_sum_threshold ${thres_sum} --remove_assistant_turns true \
    --score_heads "informative_score" \
    --input_dir test/arl_scout/p1.02_main1/all-images --frame_fps 1 \
    --output_fname ${output_dir}/eval/arl_scout_test-random_prompt-pred.json
# wait

# --------------------
# evaluate
# --------------------
# python -u -m test.evaluate --func hisum \
#     --pred_file ${output_dir}/eval/hisum_test-random_prompt-pred.json \
#     --gold_file /data/yt8m/annotations/mr_hisum.h5  \
#     --output_file ${output_dir}/eval/hisum_test-random_prompt-eval.json \
#     --alpha 0.0 --beta 1.55555 --epsilon 1.0
    # > ${output_dir}/eval/hisum_test-random_prompt-eval.log 2>&1 &