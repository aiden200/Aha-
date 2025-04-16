
output_dir=outputs/tvsum_eval_aha


# --------------------
# evaluate
# --------------------
python -u -m test.visualize \
    --pred_file ${output_dir}/eval/tvsum_test-random_prompt-pred.json \
    --gold_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv \
    --dataset tvsum \
    --max_show 5\
