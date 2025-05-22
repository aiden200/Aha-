
tvsum_output_dir=outputs/tvsum/eval
tvsum_degraded_output_dir=outputs/tvsum_degraded/eval
charades_output_dir=outputs/charades/eval
hisum_output_dir=outputs/hisum/eval


# hisum
# python -u -m test.grid_search --test_dataset hisum \
#     --pred_file ${hisum_output_dir}/hisum_test-random_prompt-pred.json \
#     --gold_file /data/yt8m/annotations/mr_hisum.h5 

    
# tvsum
# python -u -m test.grid_search --test_dataset tvsum \
#     --pred_file ${tvsum_output_dir}/tvsum_test-random_prompt-pred.json \
#     --gold_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv

# tvsum_degraded 
# python -u -m test.grid_search --test_dataset tvsum_degraded \
#     --pred_file ${tvsum_degraded_output_dir}/tvsum_test-random_prompt-pred.json \
#     --gold_file datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv


# charades
# python -u -m test.grid_search --test_dataset charades \
#     --pred_file ${charades_output_dir}/charades_test-random_prompt-pred.json \
#     --gold_file datasets/charades/annotations/test-random_prompt.json 

