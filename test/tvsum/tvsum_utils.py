from collections import defaultdict
import numpy as np
import csv
import os
import numpy as np
from sklearn.metrics import average_precision_score

def compute_map(y_true, y_scores):
    """
    Compute Mean Average Precision (mAP)
    :param y_true: Relevance scores ranging between 0 and 1
    :param y_scores: Model-generated scores for each frame/segment
    :return: Mean Average Precision (mAP)
    """
    return average_precision_score(y_true, y_scores)

def compute_top5_map(y_true, y_scores):
    """
    Compute Top-5 Mean Average Precision (mAP)
    :param y_true: Relevance scores ranging between 0 and 1
    :param y_scores: Model-generated scores for each frame/segment
    :return: Top-5 mAP
    """
    # Get indices of top-5 highest scoring predictions
    top5_indices = np.argsort(y_scores)[-5:][::-1]  # Top 5 sorted in descending order
    
    # Extract top-5 relevance scores and their predicted scores
    top5_y_true = np.array([y_true[i] for i in top5_indices])
    top5_y_scores = np.array([y_scores[i] for i in top5_indices])
    
    # Compute precision at each rank (cumulative precision)
    precisions = [np.mean(top5_y_true[:k+1]) for k in range(len(top5_y_true))]
    
    # Compute Top-5 mAP as the mean of the precision scores weighted by relevance
    top5_map = np.sum(precisions * top5_y_true) / np.sum(top5_y_true) if np.sum(top5_y_true) > 0 else 0.0
    
    return top5_map


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

def evaluate_tvsum(gt_dict, pred_dict, k=5):
    """
    Evaluate TVSum mAP using ground truth and predicted saliency scores.
    
    Parameters:
    - gt_dict: dictionary mapping video ids to ground truth scores. Each value can be:
               a 1D array (num_clips,) for single annotation, or
               a 2D array (num_clips, num_annotations) for multiple annotations.
    - pred_dict: dictionary mapping video ids to predicted clip-level saliency scores (1D array).
    - k: number of top clips to consider for AP calculation (default is 5).
    
    Returns:
    - mAP: the mean average precision across videos.
    """
    ap_scores = []

    for video_id, gt_scores in gt_dict.items():
        pred_scores = pred_dict[video_id]
        
        if gt_scores.shape[0] != pred_scores.shape[0]:
            raise ValueError(f"Number of clips mismatch for video {video_id}")

        # Rank the clips based on predicted saliency (in descending order)
        sorted_indices = np.argsort(-pred_scores)

        # Case 1: Single annotation (1D array)
        if gt_scores.ndim == 1:
            threshold = np.median(gt_scores)
            gt_binary = (gt_scores > threshold).astype(float)
            ap = compute_ap(gt_binary, sorted_indices, k)
            ap_scores.append(ap)
        # Case 2: Multiple annotations (2D array: num_clips x num_annotations)
        elif gt_scores.ndim == 2:
            num_annotations = gt_scores.shape[1]
            ap_video = []
            for i in range(num_annotations):
                gt_anno = gt_scores[:, i]
                threshold = np.median(gt_anno)
                gt_binary = (gt_anno > threshold).astype(float)
                ap_video.append(compute_ap(gt_binary, sorted_indices, k))
            ap_scores.append(np.mean(ap_video))
        else:
            raise ValueError("Ground truth scores must be either a 1D or 2D numpy array.")

    mAP = np.mean(ap_scores)
    return mAP


def compute_ap_top5(gt_binary, pred_scores, k=5):
    """
    Compute the average precision (AP) using the top k ranked clips.
    
    Parameters:
    - gt_binary: a 1D numpy array of binary ground truth labels.
    - pred_scores: a 1D numpy array of predicted saliency scores.
    - k: number of top clips to consider (default is 5).
    
    Returns:
    - ap: average precision value computed using the top k clips.
    """
    # Get indices that sort the predictions in descending order.
    sorted_indices = np.argsort(-pred_scores)
    # Select top-k ground truth labels based on prediction ranking.
    topk_labels = gt_binary[sorted_indices][:k]
    
    # If there are no positive samples in top-k, return AP as 0.
    num_pos = np.sum(topk_labels)
    if num_pos == 0:
        return 0.0
    
    hits = 0.0
    ap = 0.0
    rec_prev = 0.0
    prec_prev = 1.0
    
    for j, label in enumerate(topk_labels):
        hits += label
        rec = hits / num_pos
        prec = hits / (j + 1)
        # Use trapezoidal integration between precision and recall values.
        ap += (rec - rec_prev) * (prec + prec_prev) / 2.0
        rec_prev = rec
        prec_prev = prec
        
    return ap


def evaluate_top5_mAP(gt_dict, pred_dict, k=5):
    """
    Evaluate Top-5 mAP on the TVSum benchmark.
    
    Parameters:
    - gt_dict: dictionary mapping video IDs to ground truth scores.
      Each value can be:
        * a 1D array (num_clips,) for a single annotation, or
        * a 2D array (num_clips, num_annotations) for multiple annotations.
    - pred_dict: dictionary mapping video IDs to predicted clip-level saliency scores (1D array).
    - k: number of top clips to consider (default is 5).
    
    Returns:
    - mAP: mean Average Precision computed over videos using the top k clips.
    """
    ap_scores = []
    
    for video_id, gt_scores in gt_dict.items():
        pred_scores = pred_dict[video_id]
        
        if gt_scores.shape[0] != pred_scores.shape[0]:
            raise ValueError(f"Number of clips mismatch for video {video_id}")
        
        # Case 1: Single annotation (1D array)
        if gt_scores.ndim == 1:
            threshold = np.median(gt_scores)
            gt_binary = (gt_scores > threshold).astype(float)
            ap = compute_ap_top5(gt_binary, pred_scores, k)
            ap_scores.append(ap)
        # Case 2: Multiple annotations (2D array: num_clips x num_annotations)
        elif gt_scores.ndim == 2:
            num_annotations = gt_scores.shape[1]
            ap_per_video = []
            for i in range(num_annotations):
                gt_anno = gt_scores[:, i]
                threshold = np.median(gt_anno)
                gt_binary = (gt_anno > threshold).astype(float)
                ap = compute_ap_top5(gt_binary, pred_scores, k)
                ap_per_video.append(ap)
            ap_scores.append(np.mean(ap_per_video))
        else:
            raise ValueError("Ground truth scores must be either a 1D or 2D numpy array.")
    
    mAP = np.mean(ap_scores)
    return mAP