import json
import h5py
import numpy as np
from itertools import product
from tqdm import tqdm
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
    score = results["f1"]
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
    best_score = -np.inf 
    ground_truths = None
    
    if args.dataset == "hisum":
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)
        
    elif args.dataset == "tvsum":
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)
        
        ground_truths = get_annos(args.gold_file)
    
    total_combinations = (
        len(param_grid["alpha"]) *
        len(param_grid["beta"]) *
        len(param_grid["epsilon"])
    )
    
    NUM_WORKERS = 20
    
    param_combos = list(product(param_grid["alpha"], param_grid["beta"], param_grid["epsilon"]))
    args_list = [(alpha, beta, epsilon, predictions, args.dataset, args.gold_file, ground_truths) for alpha, beta, epsilon in param_combos]

    best_score = float("-inf")
    best_params = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(score_worker, args_list), total=len(args_list), desc=f"Grid Search {args.dataset}"))

    for score, params in results:
        if score > best_score:
            best_score = score
            best_params = params

    best_params["best_score"] = best_score
        
    return best_params

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="The type of dataset used")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to predictions JSON file.")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to gold standard HDF5 file.")
    args = parser.parse_args()

    param_grid = {
        "alpha": np.linspace(0.0, 2.0, 20), # Importance
        "beta": np.linspace(0.0, 2.0, 20), # Relevance
        "epsilon": np.linspace(0.0, 1.0, 11) # Uncertainty
    }

    best_params = grid_search(args, param_grid)
    
    print("Best parameters found:")
    print(best_params)