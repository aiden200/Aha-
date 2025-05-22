
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
from .datasets import FastAndAccurateStreamingVideoQADataset
from .inference import LiveInferForBenchmark, DoNothingDataCollator, round_numbers
from .evaluate import normalize_pred_list, is_time_in_span, calculate_iou, qvh_to_charades_format
import concurrent.futures
from .hisum.hisum_eval import hisum_evaluate_scores
from .tvsum.tvsum_utils import *


def score_worker(args_tuple):
    alpha, beta, epsilon, uncertainty_threshold, predictions, dataset, hdf_path, ground_truths = args_tuple

    if dataset == "hisum":
        with h5py.File(hdf_path, "r") as hdf:
            score = hisum_score_calculation(predictions, hdf, alpha, beta, epsilon, uncertainty_threshold)

    elif dataset == "tvsum" or dataset == "tvsum_degraded":
        score = tvsum_score_calculation(predictions, ground_truths, alpha, beta, epsilon, uncertainty_threshold)
    
    elif dataset == "charades":
        score = charades_eval(predictions, ground_truths, alpha, beta, epsilon, uncertainty_threshold)

    return score, {"alpha": alpha, "beta": beta, "epsilon": epsilon, "uncertainty_threshold": uncertainty_threshold}



def hisum_score_calculation(predictions, hdf, alpha, beta, epsilon, uncertainty_threshold):
    
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
            curr_pred_score = alpha * e["informative_score"] + beta * e['relevance_score']
            if e["uncertainty_score"] >= uncertainty_threshold:
                diff = e["uncertainty_score"] - uncertainty_threshold
                penalty = diff * epsilon
                curr_pred_score -= penalty
            pred_scores.append(curr_pred_score)

            ground_truth_frame_scores.append(vid_ground_truth[i-1])
        pred_scores = np.array(pred_scores)
        ground_truth_frame_scores = np.array(ground_truth_frame_scores)
        
        pred_dict[video_uuid] = pred_scores
        gt_dict[video_uuid] = ground_truth_frame_scores
    
    spearman_kendall = False
    results = hisum_evaluate_scores(gt_dict, pred_dict, spearman_kendall=spearman_kendall, print_logs=False)
    if spearman_kendall:
        kendall = results["kendall"] 
        spearman = results["spearman"]
    map50 = results["mAP@50"] 
    map15 = results["mAP@15"] 
    f1 = results["f1"]
    score = map50
    return score


def tvsum_score_calculation(predictions, ground_truths, alpha, beta, epsilon=None, uncertainty_threshold=None):
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

            curr_pred_score = alpha * e["informative_score"] + beta * e['relevance_score']
            if e["uncertainty_score"] >= uncertainty_threshold:
                diff = e["uncertainty_score"] - uncertainty_threshold
                penalty = diff * epsilon
                curr_pred_score -= penalty
            pred_scores.append(curr_pred_score)

            ground_truth_frame_scores.append(vid_ground_truth[true_frame])

        
        pred_scores = np.array(pred_scores)
        ground_truth_frame_scores = np.array(ground_truth_frame_scores)
        
        pred_dict[video_uuid] = pred_scores
        gt_dict[video_uuid] = ground_truth_frame_scores

    mAP50, mAP15, top_5_mAP, spearman, kendall = evaluate_tvsum(gt_dict, pred_dict)
    f115 = evaluate_f1(gt_dict, pred_dict)
    # score = mAP50
    score = top_5_mAP
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




def charades_eval(predictions, ground_truths, alpha, beta, epsilon, uncertainty_threshold):
    iou_scores_list_dict = {threshold: list() for threshold in np.arange(0.30, 0.71, 0.02)}
    for pred_example in predictions:
        gold_example = ground_truths[pred_example['question_id']]
        video_times, pred_scores = list(), list()
        for e in pred_example['debug_data']:
            video_times.append(e['time'])
            if 'relevance_score' in e:
                curr_pred_score = alpha * e["informative_score"] + beta * e['relevance_score']
                if e["uncertainty_score"] >= uncertainty_threshold:
                    diff = e["uncertainty_score"] - uncertainty_threshold
                    penalty = diff * epsilon
                    curr_pred_score -= penalty
                pred_scores.append(curr_pred_score)
            else:
                pred_scores.append(0)

        pred_scores = normalize_pred_list(pred_scores)
        gold_scores = [is_time_in_span(time, gold_example['timestamps']) for time in video_times]
        for threshold in iou_scores_list_dict:
            iou = calculate_iou(pred_scores, gold_scores, threshold, debug_data=pred_example['question_id'])
            iou_scores_list_dict[threshold].append(iou)
    

    for threshold in iou_scores_list_dict:
        mean_iou = np.mean(iou_scores_list_dict[threshold]) * 100
        recall_0_3 = np.mean([e >= 0.3 for e in iou_scores_list_dict[threshold]]) * 100
        recall_0_5 = np.mean([e >= 0.5 for e in iou_scores_list_dict[threshold]]) * 100
        recall_0_7 = np.mean([e >= 0.7 for e in iou_scores_list_dict[threshold]]) * 100
        # print(f'score threshold = {threshold:.2f}: {mean_iou:.2f}/{recall_0_3:.2f}/{recall_0_5:.2f}/{recall_0_7:.2f}')

    best_among_all_thres = [max([iou_list[i] for iou_list in iou_scores_list_dict.values()]) for i in range(len(predictions))]
    mean_iou = np.mean(best_among_all_thres) * 100
    recall_0_3 = np.mean([e >= 0.3 for e in best_among_all_thres]) * 100
    recall_0_5 = np.mean([e >= 0.5 for e in best_among_all_thres]) * 100
    recall_0_7 = np.mean([e >= 0.7 for e in best_among_all_thres]) * 100


    # print(f'best among all thresholds: {mean_iou:.2f}/{recall_0_3:.2f}/{recall_0_5:.2f}/{recall_0_7:.2f}')
    return recall_0_5




        


def grid_search(args, param_grid, save_path, uncertainty=True):

    best_params = {"alpha": None, "beta": None, "epsilon": None, "uncertainty_threshold": None}
    hdf = None
    ground_truths = None
    uncertainty_threshold = None
    

    with open(save_path, "r") as f:
        best_args_json = json.load(f)


    
    if args.test_dataset == "hisum":
        
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)
        
    elif args.test_dataset == "tvsum" or args.test_dataset == "tvsum_degraded":
        
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)
        
        ground_truths = get_annos(args.gold_file)
    
    
    elif args.test_dataset == "charades":
        predictions = [json.loads(line) for line in open(args.pred_file)]
        ground_truths = json.load(open(args.gold_file))
        if 'answer' in ground_truths[0] and 'saliency_scores' in ground_truths[0]['answer']:
            # this is a qvh dataset, convert it to charades format
            ground_truths = [qvh_to_charades_format(e) for e in ground_truths]
        ground_truths = {e['question_id']: e for e in ground_truths}
    
    total_combinations = (
        len(param_grid["alpha"]) *
        len(param_grid["beta"]) *
        len(param_grid["epsilon"])
    )
    
    NUM_WORKERS = 100
    if uncertainty:
        param_combos = list(product(param_grid["alpha"], param_grid["beta"], param_grid["epsilon"], param_grid["uncertainty_threshold"]))
        args_list = [(alpha, beta, epsilon, uncertainty_threshold, predictions, args.test_dataset, args.gold_file, ground_truths) for alpha, beta, epsilon, uncertainty_threshold in param_combos]
    else:
        param_combos = list(product(param_grid["alpha"], param_grid["beta"], param_grid["epsilon"]))
        args_list = [(alpha, beta, epsilon, uncertainty_threshold, predictions, args.test_dataset, args.gold_file, ground_truths) for alpha, beta, epsilon in param_combos]

    

    best_score = float("-inf")
    best_params = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(score_worker, args_list), total=len(args_list), desc=f"Grid Search {args.test_dataset}"))

    for score, params in results:
        if score > best_score:
            best_score = score
            best_params = params

    # best_params["best_score"] = best_score

    print(f"Best score: {best_score}")

    best_args_json[args.test_dataset] = best_params

    with open(save_path, "w") as f:
        json.dump(best_args_json, f)
        
    return best_params

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", type=str, required=False)
    parser.add_argument("--gold_file", type=str, required=False)
    parser.add_argument("--pred_file", type=str, required=False)
    parser.add_argument("--output_fname", type=str, required=False)
    # parser.add_argument("--output_dir", type=str, required=False)


    args = parser.parse_args()
    save_path = "outputs/grid_search_params.json"
    non_eval_metrics = ["hisum", "tvsum", "tvsum_degraded", "charades"]

    if args.test_dataset in non_eval_metrics:
        param_grid = {
            "alpha": np.linspace(0.0, 2.0, 10), # Importance
            "beta": np.linspace(-1.0, 2.0, 15), # Relevance
            "epsilon": np.linspace(-5, 5, 15), # Uncertainty
            # "alpha": np.linspace(1.47, 1.47, 1), # Importance
            # "beta": np.linspace(1.87, 1.87, 1), # Relevance
            "uncertainty_threshold": np.linspace(0.04, 0.15, 10) # Threshold
        }
        best_params = grid_search(args, param_grid, save_path, True)
    elif args.test_dataset =="youcook2":
        raise ValueError("Deprecated")
        # param_grid = {
        #     "threshold": np.linspace(1, 3, 10)
        # }
        # best_params = grid_search_with_inference(args, param_grid)
    
    print("Best parameters found:")
    print(best_params)


