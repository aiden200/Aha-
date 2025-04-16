output_dir=outputs/hisum_eval
# pretrained_dir=outputs/mmduet
pretrained_dir=aiden200/aha
mkdir -vp  ${output_dir}/eval

if [ -f "${output_dir}/eval/hisum_test-random_prompt-pred.log" ]; then
    rm "${output_dir}/eval/hisum_test-random_prompt-pred.log"
fi


# # --------------------
# # run inference
# # --------------------
python -u -m test.inference --grounding_mode true \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --test_dataset hisum \
    --skip_eval false \
    --video_metadata_file datasets/hisum/videos_metadata.json \
    --caption_metadata_file datasets/hisum/annotations/mr_hisum_metadata.csv \
    --hisum_h5_file /data/yt8m/annotations/mr_hisum.h5 \
    --anno_file datasets/hisum/annotations/split.json \
    --lora_pretrained ${pretrained_dir}  \
    --stream_end_prob_threshold 1 \
    --input_dir /data/yt8m/videos/ --frame_fps 1 --max_num_frames 400 \
    --test_fname datasets/hisum/annotations/test-random_prompt.json \
    --output_fname ${output_dir}/eval/hisum_test-random_prompt-pred.json \
    > ${output_dir}/eval/hisum_test-random_prompt-pred.log 
# wait

    # --caption_metadata_file datasets/hisum/ydata-hisum50-v1_1/data/ydata-hisum50-info.tsv \

# --------------------
# evaluate
# --------------------
python -u -m test.evaluate --func hisum \
    --pred_file ${output_dir}/eval/hisum_test-random_prompt-pred.json \
    --gold_file /data/yt8m/annotations/mr_hisum.h5  \
    --output_file ${output_dir}/eval/hisum_test-random_prompt-eval.json \
    --alpha 0.3 --beta 0.1 --epsilon 0.0 \
    > ${output_dir}/eval/hisum_test-random_prompt-eval.log 2>&1 &