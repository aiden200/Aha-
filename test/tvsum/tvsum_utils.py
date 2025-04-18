from collections import defaultdict
import numpy as np
import csv
import os
import numpy as np
from sklearn.metrics import average_precision_score

def binarize_gt(gt_scores, rho):
    n = len(gt_scores)
    k = max(1, int(rho * n))
    thresh = np.sort(gt_scores)[-k]
    return (gt_scores >= thresh).astype(int)

def map_at_rho(gt_scores, pred_scores, rho):
    gt_bin = binarize_gt(gt_scores, rho)
    return average_precision_score(gt_bin, pred_scores)


def evaluate_tvsum(gt_dict, pred_dict):

    map50_scores = []
    map15_scores = []

    for video_id, gt_scores in gt_dict.items():
        pred_scores = pred_dict[video_id]


        map50 = map_at_rho(gt_scores, pred_scores, rho=0.50)
        map15 = map_at_rho(gt_scores, pred_scores, rho=0.15)
        map50_scores.append(map50)
        map15_scores.append(map15)

    mAP50 = np.mean(map50_scores)
    mAP15 = np.mean(map15_scores)
    return mAP50, mAP15


import numpy as np
from sklearn.metrics import f1_score

def f1_at_rho(gt_scores, pred_scores, rho):
    """
    Compute F1 score by selecting the top-rho fraction of frames as positives,
    for both ground truth and predictions.

    :param gt_scores: 1D array of continuous ground-truth importance scores
    :param pred_scores: 1D array of continuous predicted scores
    :param rho: fraction of frames to select (e.g. 0.15 or 0.50)
    :return: F1 score
    """
    n = len(gt_scores)
    k = max(1, int(rho * n))

    # Binarize GT: top-k frames become positives
    thresh_gt = np.sort(gt_scores)[-k]
    gt_bin = (gt_scores >= thresh_gt).astype(int)
    

    # Binarize predictions: top-k predicted as positives
    topk_pred = np.argsort(pred_scores)[-k:]
    pred_bin = np.zeros(n, dtype=int)
    pred_bin[topk_pred] = 1

    # Compute F1
    return f1_score(gt_bin, pred_bin)

def evaluate_f1(gt_dict, pred_dict, rho=0.15):
    """
    Compute average F1 at rho over a set of videos.
    :param gt_dict: dict video_id -> gt_scores array
    :param pred_dict: dict video_id -> pred_scores array
    :param rho: fraction of frames to select
    :return: mean F1 across all videos
    """
    f1_list = []
    for vid, gt_scores in gt_dict.items():
        pred_scores = pred_dict[vid]
        f1_list.append(f1_at_rho(gt_scores, pred_scores, rho))
    return np.mean(f1_list)



def get_annos(annotation_file) -> dict:
        # load tsv file and get average importance scores
        assert os.path.exists(annotation_file), f"Error, {annotation_file} does not exist"
        vid_count = defaultdict(int)
        annotations = {}
        with open(annotation_file, 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                video_id = row[0]
                category_code = row[1]
                importance_scores = np.array(list(map(int, row[2].split(','))), dtype=np.float64)
                if video_id not in annotations:
                    annotations[video_id] = {
                        "importance_scores": importance_scores,
                        "video_uid": video_id,
                        "category_code": category_code
                    }
                else:
                    annotations[video_id]["importance_scores"]  += importance_scores
                
                vid_count[video_id] += 1
        
        for video in annotations:
            annotations[video]["importance_scores"] /= vid_count[video]
            # Normalizing the score, the maximum score is 5
            annotations[video]["importance_scores"]  /= 5.0
            annotations[video]["importance_scores"] = annotations[video]["importance_scores"].tolist()
        return annotations




def iou_1d(pred_interval, gt_interval):
    """
    Computes IoU of two 1D intervals: pred_interval = (start_p, end_p)
                                       gt_interval = (start_g, end_g)
    """
    start_p, end_p = pred_interval
    start_g, end_g = gt_interval

    # Compute overlap
    overlap_start = max(start_p, start_g)
    overlap_end = min(end_p, end_g)
    overlap = max(0, overlap_end - overlap_start)

    # Compute union
    union = (end_p - start_p) + (end_g - start_g) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def compute_recall_at_1(ground_truths, predictions, iou_threshold=0.5):
    """
    :param ground_truths: list of ground-truth intervals, one per sample
    :param predictions: list of lists, each inner list has (interval, score)
                        e.g. predictions[i] = [((s1, e1), score1), ((s2, e2), score2), ...]
    :param iou_threshold: IoU threshold (e.g., 0.5 or 0.7)
    :return: recall_at_1 (float)
    """
    hits = 0
    total = len(ground_truths)

    for i, gt_interval in enumerate(ground_truths):
        # Sort by descending score
        preds = sorted(predictions[i], key=lambda x: x[1], reverse=True)
        # Take the top 1 predicted interval
        best_pred, best_score = preds[0]

        if iou_1d(best_pred, gt_interval) >= iou_threshold:
            hits += 1

    return hits / total


import numpy as np

def compute_map_at_iou(ground_truths, predictions, iou_threshold):
    """
    :param ground_truths: list of ground-truth intervals (or multiple intervals) for each sample
    :param predictions: list of lists of (interval, score) for each sample
    :param iou_threshold: IoU threshold
    :return: mAP at that IoU threshold
    """
    # Gather all predictions in a single list to rank them globally
    # Each entry: (score, is_positive)
    score_label_pairs = []

    for i, gt_interval in enumerate(ground_truths):
        # Could handle multiple GT intervals per sample if needed.
        # For simplicity, assume one GT interval per sample.

        for (pred_interval, score) in predictions[i]:
            iou_val = iou_1d(pred_interval, gt_interval)
            is_positive = (iou_val >= iou_threshold)
            score_label_pairs.append((score, is_positive))

    # Sort all predictions in descending score
    score_label_pairs.sort(key=lambda x: x[0], reverse=True)

    # Compute precision/recall points
    y_true = [1 if x[1] else 0 for x in score_label_pairs]
    y_scores = [x[0] for x in score_label_pairs]

    mAP = average_precision_score(y_true, y_scores)

    return mAP

def compute_average_map(ground_truths, predictions, iou_thresholds):
    """
    :param iou_thresholds: list of IoU thresholds, e.g. [0.5, 0.75] or np.arange(0.5,1.0,0.05)
    :return: average of mAP across all IoU thresholds
    """
    maps = []
    for thresh in iou_thresholds:
        maps.append(compute_map_at_iou(ground_truths, predictions, thresh))
    return np.mean(maps)


def compute_iou_50_and_75(y_truth, y_pred):
    # For MR:
    mAP_50 = compute_map_at_iou(y_truth, y_pred, iou_threshold=0.5)
    mAP_75 = compute_map_at_iou(y_truth, y_pred, iou_threshold=0.75)
    average_map = compute_average_map(y_truth, y_pred,
                                    iou_thresholds=np.arange(0.5, 1.0, 0.05))
    print(f"mAP at IoU=0.5: {mAP_50:.4f}")
    print(f"mAP at IoU=0.75: {mAP_75:.4f}")
    print(f"Average mAP over 0.5:0.05:0.95 => {average_map:.4f}")


def compute_ap(gt_binary, sorted_indices, k=5):
    """
    Compute average precision (AP) for one video.
    
    Parameters:
    - gt_binary: a 1D numpy array of binary ground truth labels (after thresholding)
    - sorted_indices: indices that sort the predicted scores in descending order
    - k: number of top clips to consider (default is 5)
    
    Returns:
    - ap: average precision value for the video
    """
    # Select the top-k ground truth labels according to the prediction ranking.
    selected = gt_binary[sorted_indices][:k]
    num_gt = np.sum(selected)
    if num_gt == 0:
        return 0.0

    hits = 0
    ap = 0.0
    rec_prev = 0.0
    prec_prev = 1.0
    for j, label in enumerate(selected):
        hits += label
        rec = hits / num_gt
        prec = hits / (j + 1)
        # Trapezoidal integration to compute area under the precision-recall curve
        ap += (rec - rec_prev) * (prec + prec_prev) / 2.0
        rec_prev = rec
        prec_prev = prec
    return ap
