
output_dir_hisum=outputs/hisum_eval


# --------------------
# evaluate
# --------------------
python -u -m test.visualize \
    --pred_file ${output_dir_hisum}/eval/hisum_test-random_prompt-pred.json \
    --gold_file /data/yt8m/annotations/mr_hisum.h5  \
    --dataset hisum \
    --alpha 0.0 --beta 1.368 --epsilon 0.9 \
    --max_show 5\
