
output_dir=outputs/grid_search_results
tvsum_output_dir=outputs/tvsum_eval_aha/eval
hisum_output_dir=outputs/hisum_eval/eval
mkdir -vp ${output_dir}

if [ -f '${output_dir}/grid_search.log' ]; then
    rm '${output_dir}/grid_search.log'
fi

# --------------------
# evaluate hisum
# --------------------
python -u -m test.grid_search --dataset hisum \
    --pred_file ${hisum_output_dir}/hisum_test-random_prompt-pred.json \
    --gold_file /data/yt8m/annotations/mr_hisum.h5  \
    
# --------------------
# evaluate tvsum
# --------------------
python -u -m test.grid_search --dataset tvsum \
    --pred_file ${tvsum_output_dir}/tvsum_test-random_prompt-pred.json \
    --gold_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv

