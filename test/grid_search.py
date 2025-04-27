import json
import h5py
import numpy as np
from itertools import product
from tqdm import tqdm

import os
from tqdm import tqdm
import numpy as np
import torch
import transformers
import h5py
import yaml
import re


logger = transformers.logging.get_logger('inference')

from models import parse_args
from .dvc.eval_dvc import eval_with_files 
from .qvh.eval import eval_submission, load_jsonl
from .datasets import FastAndAccurateStreamingVideoQADataset
from .inference import LiveInferForBenchmark, DoNothingDataCollator, round_numbers
import concurrent.futures
from .hisum.hisum_eval import hisum_evaluate_scores
from .tvsum.tvsum_utils import *


def score_worker(args_tuple):
    alpha, beta, epsilon, predictions, dataset, hdf_path, ground_truths = args_tuple

    if dataset == "hisum":
        with h5py.File(hdf_path, "r") as hdf:
            score = hisum_score_calculation(predictions, hdf, alpha, beta, epsilon)

    elif dataset == "tvsum":
        score = tvsum_score_calculation(predictions, ground_truths, alpha, beta, epsilon)
    
    elif dataset == "qvh":
        score = qvh_eval(predictions, ground_truths, alpha, beta, epsilon)

    return score, {"alpha": alpha, "beta": beta, "epsilon": epsilon}



def hisum_score_calculation(predictions, hdf, alpha, beta, epsilon):
    
    gt_dict = {}
    pred_dict = {}
    for prediction in predictions:
        video_uuid = prediction["video_uuid"]
        h5_video_identifier = prediction["h5_identifier"]
        vid_ground_truth = list(hdf[h5_video_identifier]["gtscore"])
        ground_truth_frame_scores = []
        pred_scores = list()
        for i in range(1, min(len(prediction['debug_data']), len(vid_ground_truth))):
            e = prediction['debug_data'][i]
            pred_scores.append(
                alpha *e["informative_score"]\
                    + beta * e['relevance_score'] \
                        + epsilon * e["uncertainty_score"])
            ground_truth_frame_scores.append(vid_ground_truth[i-1])
        pred_scores = np.array(pred_scores)
        ground_truth_frame_scores = np.array(ground_truth_frame_scores)
        
        pred_dict[video_uuid] = pred_scores
        gt_dict[video_uuid] = ground_truth_frame_scores
    

    results = hisum_evaluate_scores(gt_dict, pred_dict, print_logs=False)
    score = results["mAP@50"]
    return score


def tvsum_score_calculation(predictions, ground_truths, alpha, beta, epsilon):
    gt_dict = {}
    pred_dict = {}
    for prediction in predictions:
        video_uuid = prediction["video_uuid"]
        true_frames_list = prediction['true_frames_list']
        vid_ground_truth = ground_truths[video_uuid]["importance_scores"] #counterintuitative but this is relevance score
        ground_truth_frame_scores = []
        pred_scores = list()
        for i in range(len(prediction['debug_data'])):
            e = prediction['debug_data'][i]
            true_frame = true_frames_list[i]
            # video_times.append(e['video_time'])
            pred_scores.append(alpha * e["informative_score"] \
                + beta * e['relevance_score'] \
                    + epsilon * e["uncertainty_score"])

            ground_truth_frame_scores.append(vid_ground_truth[true_frame])

        
        pred_scores = np.array(pred_scores)
        ground_truth_frame_scores = np.array(ground_truth_frame_scores)
        
        pred_dict[video_uuid] = pred_scores
        gt_dict[video_uuid] = ground_truth_frame_scores

    mAP50, mAP15 = evaluate_tvsum(gt_dict, pred_dict)
    f115 = evaluate_f1(gt_dict, pred_dict)
    score = mAP50
    return score


def youcook2_eval(test_args, pred_examples, gold_examples, gold_file, pred_file):
    pred_out, gold_out = dict(), list()
    for pred_example in pred_examples:
        if test_args.is_online_model:
            captions, prev_sent, start_time, end_time = list(), None, None, None
            for turn in pred_example['model_response_list']:
                if turn['role'] == 'user': continue
                if turn['content'] != prev_sent:
                    if start_time is not None:
                        captions.append({'timestamp': [start_time, end_time], 'caption': prev_sent})
                    prev_sent, start_time, end_time = turn['content'], end_time, turn['time']
                else:
                    end_time = turn['time']

            if start_time is not None:
                captions.append({'timestamp': [start_time, end_time], 'caption': prev_sent})
            pred_out[str(pred_example['question_id'])] = captions
        else:
            model_response = pred_example['model_response'][0] if isinstance(pred_example['model_response'], list) else pred_example['model_response']
            if 'vtimellm' in pred_file:
                # this is a vtimellm format response
                pattern = r"From (\d+) to (\d+), (.*)"
                matches = re.findall(pattern, model_response)
                captions = list()
                video_length = pred_example['video_duration']
                for match in matches:
                    captions.append({'timestamp': [int(match[0]) / 100 * video_length, int(match[1]) / 100 * video_length], 'caption': match[2]})
                pred_out[str(pred_example['question_id'])] = captions
            else:
                # this is a timechat format response
                pattern = r"(\d+\.\d+) - (\d+\.\d+)\s*seconds,\s*(.*)"
                # pattern = r"(\d+\.\d+)\s*seconds,\s*(.*)"
                matches = re.findall(pattern, model_response)
                captions = list()
                for match in matches:
                    start_time, end_time, action = float(match[0]), float(match[1]), match[2]
                    captions.append({'timestamp': [start_time, end_time], 'caption': action})
                    start_time = end_time
                pred_out[str(pred_example['question_id'])] = captions

    # youcook2 dense captioning evaluation
    for gold_example in gold_examples:
        if str(gold_example['question_id']) not in pred_out: continue
        segments = [turn['time'] for turn in gold_example['answer']]
        answer_list = [turn['content'] for turn in gold_example['answer']]
        answer_list = [ans.replace('. ', ', ') for ans in answer_list]
        pure_cap = '. '.join(answer_list)
        gold_out.append({'image_id': str(gold_example['question_id']), 'segments': segments, 'pure_cap': pure_cap})

    base_folder = os.path.dirname(pred_file)
    os.makedirs(os.path.join(base_folder, 'tmp'), exist_ok=True)
    temp_pred_fname = os.path.join(base_folder, f'tmp/{os.path.basename(pred_file)}')
    temp_gold_fname = os.path.join(base_folder, 'tmp/val.json')
    with open(temp_pred_fname, 'w') as f:
        json.dump(pred_out, f)
    with open(temp_gold_fname, 'w') as f:
        json.dump({"annotations": gold_out}, f)
    results = eval_with_files(temp_pred_fname, temp_gold_fname)
    return results


def qvh_eval(predictions, ground_truths, alpha, beta, epsilon):
    reformatted_pred_list = list()

    for example in predictions:
        
        frame_interval = example['debug_data'][1]['time'] - example['debug_data'][0]['time']
        two_sec_frames = int(2 / frame_interval)
        video_times, pred_scores = list(), list()
        for e in example['debug_data']:
            video_times.append(e['time'])
            if 'relevance_score' in e:
                # pred_scores.append(e['relevance_score'][1])
                pred_scores.append(
                    alpha* e['informative_score'] \
                        + beta * e['relevance_score']\
                            + epsilon * e['uncertainty_score'])
            else:
                pred_scores.append(0)

        pred_saliency_scores = [sum(pred_scores[i: i + two_sec_frames]) for i in range(0, len(pred_scores), two_sec_frames)]
        reformatted_pred_list.append({'qid': example['question_id'], 'pred_saliency_scores': pred_saliency_scores})

    results = eval_submission(reformatted_pred_list, ground_truths, verbose=False, match_number=False)
    score = results['HL-min-Fair']['HL-mAP']
    return score


def grid_search_with_inference(args, param_grid):

    with open("paths.yaml", "r") as f:
        all_args = yaml.safe_load(f)
        yc_args = all_args["youcook2"]
        pretrained_args = all_args["pretrained_args"]
    os.environ['RANK'] = "0"
    test_args = parse_args('test')

    test_args.lora_pretrained = pretrained_args["lora_pretrained"]
    test_args.llm_pretrained = pretrained_args["llm_pretrained"]
    test_args.bf16 = pretrained_args["bf16"]

    test_args.test_dataset = yc_args["test_dataset"]
    test_args.input_dir = yc_args["input_dir"]
    test_args.frame_fps = yc_args["frame_fps"]
    test_args.max_num_frames = yc_args["max_num_frames"]
    test_args.test_fname = yc_args["test_fname"]
    test_args.stream_end_score_sum_threshold = yc_args["stream_end_score_sum_threshold"]
    test_args.remove_assistant_turns = yc_args["remove_assistant_turns"]
    test_args.start_idx = yc_args["start_idx"]
    test_args.score_heads = yc_args["score_heads"]


    with open(all_args["grid_search"]["save_path"], "r") as f:
        best_args_json = json.load(f)

    
    test_args.start_idx = 0
    test_args.end_idx = 25

    gold_file = yc_args["gold_file"]
    gs_output_file = yc_args["gridsearch_output_file"]
    gs_output_file = os.path.join(yc_args["output_dir"], "eval", gs_output_file)
 
    
    all_results = {"best_result": {
                        "best_score": float("-inf"), 
                        "best_params": None,
                        "all_scores": None
                        }, 
                    "all_params": {

                        }
                }

    
    print(test_args)
    infer = LiveInferForBenchmark(test_args)
    
        

    for threshold in param_grid["threshold"]:
        infer.reset()
        test_args.output_fname = f"{yc_args['output_dir']}/eval/youcook2_val-thres_sum_{str(threshold)}-rm_ass_turns-pred.json"


        test_args.threshold = threshold

        dataset = FastAndAccurateStreamingVideoQADataset(
                data_file=test_args.test_fname, video_base_folder=test_args.input_dir,
                start_idx=test_args.start_idx, end_idx=test_args.end_idx,
                output_fps=test_args.frame_fps, output_resolution=test_args.frame_resolution, max_num_frames=test_args.max_num_frames,
                time_instruction_format=test_args.time_instruction_format, system_prompt=test_args.system_prompt
            )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=DoNothingDataCollator())
        f_out = open(test_args.output_fname, 'w')
        
        for data_i, data in enumerate(tqdm(dataloader)):
            question_id, video_frames, conversation, fps, video_duration = data
            if question_id is None: continue
            infer.reset()
            # print(f"num frames and fps for {question_id}: {len(video_frames)}, {fps}")
            infer.set_fps(fps=fps)
            infer.input_video_stream(video_frames)
            infer.input_query_stream(conversation)
            model_response_list = infer.inference()
            res = {'question_id': question_id, 'model_response_list': model_response_list, 'video_duration': video_duration}
            res['debug_data'] = round_numbers(infer.debug_data_list, 3)
            f_out.write(json.dumps(res) + '\n')
            if data_i % 5 == 0:
                f_out.flush()
        f_out.close()


        pred_examples = [json.loads(line) for line in open(test_args.output_fname)]
        gold_examples = json.load(open(gold_file))

        results = youcook2_eval(test_args, pred_examples, gold_examples, gold_file, test_args.output_fname )
        all_results["all_params"][threshold] = results
        f1 = results["F1_Score"][0]


        if f1 > all_results["best_result"]["best_score"]:
            all_results["best_result"]["best_score"] = f1
            all_results["best_result"]["best_params"] = threshold
            all_results["best_result"]["all_scores"] = results

        print(f"Results for YouCook2 with threshold {threshold}")
        print(results)
        with open(gs_output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
    
    all_args[args.test_dataset] = threshold
    
    with open(all_args["grid_search"]["save_path"], "w") as f:
        json.dump(best_args_json, f)


        


def grid_search(args, param_grid):
    """
    Performs a grid search over the weight parameters (alpha, beta, epsilon)
    to combine informative_score, relevance_score, and uncertainty_score.

    For each weight combination, compute a weighted sum for each debug entry,
    then evaluate the predictions using hisum_evaluate_scores. The combination
    with the highest validation score (here, Pearson correlation) is stored as the best.

    Parameters:
        predictions (list): List of prediction dictionaries loaded from JSON.
        hdf (h5py.File): Opened HDF5 file object that contains ground truth data.
        param_grid (dict): Dictionary with keys 'alpha', 'beta', 'epsilon' and values being
                           iterables of candidate values.
                           e.g., {"alpha": [0.0, 0.1, ..., 1.0], "beta": [...], "epsilon": [...]}

    Returns:
        dict: A dictionary with the best parameters and the corresponding evaluation score.
              For example: {"alpha": 0.3, "beta": 0.5, "epsilon": 0.2, "pearson": 0.85}
    """

    best_params = {"alpha": None, "beta": None, "epsilon": None}
    hdf = None
    ground_truths = None


    with open("paths.yaml", "r") as f:
        dataset_args = yaml.safe_load(f)
    

    with open(dataset_args["grid_search"]["save_path"], "r") as f:
        best_args_json = json.load(f)


    dataset_output_dir = dataset_args[args.test_dataset]["output_dir"]
    args.gold_file = dataset_args[args.test_dataset]["gold_file"]
    args.pred_file = os.path.join(dataset_output_dir, dataset_args[args.test_dataset]["pred_file"]) 
    
    if args.test_dataset == "hisum":
        
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)
        
    elif args.test_dataset == "tvsum":
        
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)
        
        ground_truths = get_annos(args.gold_file)
    
    elif args.test_dataset == "qvh":
        predictions = [json.loads(line) for line in open(args.pred_file)]
        ground_truths = load_jsonl(args.gold_file)
    
    total_combinations = (
        len(param_grid["alpha"]) *
        len(param_grid["beta"]) *
        len(param_grid["epsilon"])
    )
    
    NUM_WORKERS = 100
    
    param_combos = list(product(param_grid["alpha"], param_grid["beta"], param_grid["epsilon"]))
    args_list = [(alpha, beta, epsilon, predictions, args.test_dataset, args.gold_file, ground_truths) for alpha, beta, epsilon in param_combos]

    best_score = float("-inf")
    best_params = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(score_worker, args_list), total=len(args_list), desc=f"Grid Search {args.test_dataset}"))

    for score, params in results:
        if score > best_score:
            best_score = score
            best_params = params

    best_params["best_score"] = best_score

    best_args_json[args.test_dataset] = best_params

    with open(dataset_args["grid_search"]["save_path"], "w") as f:
        json.dump(best_args_json, f)
        
    return best_params

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", type=str, required=True, help="The type of dataset used")
    parser.add_argument("--pred_file", type=str, required=False, help="Path to predictions JSON file.")
    parser.add_argument("--gold_file", type=str, required=False, help="Path to gold standard HDF5 file.")
    args = parser.parse_args()

    non_eval_metrics = ["hisum", "tvsum", "qvh"]

    if args.test_dataset in non_eval_metrics:
        param_grid = {
            "alpha": np.linspace(-0.5, 1.5, 10), # Importance
            "beta": np.linspace(-0.5, 2.0, 10), # Relevance
            "epsilon": np.linspace(-10.0, 10.0, 10) # Uncertainty
        }
        best_params = grid_search(args, param_grid)
    elif args.test_dataset =="youcook2":
        param_grid = {
            "threshold": np.linspace(2, 20, 20)
        }
        best_params = grid_search_with_inference(args, param_grid)
    
    print("Best parameters found:")
    print(best_params)