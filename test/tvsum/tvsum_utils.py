from collections import defaultdict
import csv
import os
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score

def binarize_gt(gt_scores, rho):
    n = len(gt_scores)
    k = max(1, int(rho * n))
    thresh = np.sort(gt_scores)[-k]
    return (gt_scores >= thresh).astype(int)

def map_at_rho(gt_scores, pred_scores, rho):
    gt_bin = binarize_gt(gt_scores, rho)
    return average_precision_score(gt_bin, pred_scores)

def evaluate_top5_map_tvsum(gt_dict, pred_dict, rho=0.5, top_k=5):
    ap_list = []

    for vid in gt_dict:
        gt_scores = np.array(gt_dict[vid])
        pred_scores = np.array(pred_dict[vid])

        assert len(gt_scores) == len(pred_scores), f"Length mismatch for video {vid}"

        gt_binary = binarize_gt(gt_scores, rho)

        sorted_indices = np.argsort(pred_scores)[::-1]
        ap = compute_ap(gt_binary, sorted_indices, k=top_k)
        ap_list.append(ap)

    return np.mean(ap_list)

def evaluate_tvsum(gt_dict, pred_dict):

    map50_scores = []
    map15_scores = []
    kendall_list = []
    spearman_list = []

    for video_id, gt_scores in gt_dict.items():
        pred_scores = pred_dict[video_id]
        if len(gt_scores) != len(pred_scores):
            continue  # sanity check

        if len(gt_scores) > 1:
            spearman_corr, _ = spearmanr(gt_scores, pred_scores)
            kendall_corr, _ = kendalltau(gt_scores, pred_scores)
        else:
            spearman_corr = kendall_corr = 0.0

        spearman_list.append(spearman_corr)
        kendall_list.append(kendall_corr)

        map50 = map_at_rho(gt_scores, pred_scores, rho=0.50)
        map15 = map_at_rho(gt_scores, pred_scores, rho=0.15)
        map50_scores.append(map50)
        map15_scores.append(map15)

    mAP50 = np.mean(map50_scores)
    mAP15 = np.mean(map15_scores)
    top_5_map = evaluate_top5_map_tvsum(gt_dict, pred_dict)
    spearman = np.mean(spearman_list)
    kendall = np.mean(kendall_list)
    
    return mAP50, mAP15, top_5_map, spearman, kendall




def f1_at_rho(gt_scores, pred_scores, rho):
    n = len(gt_scores)
    k = max(1, int(rho * n))

    thresh_gt = np.sort(gt_scores)[-k]
    gt_bin = (gt_scores >= thresh_gt).astype(int)
    
    topk_pred = np.argsort(pred_scores)[-k:]
    pred_bin = np.zeros(n, dtype=int)
    pred_bin[topk_pred] = 1

    return f1_score(gt_bin, pred_bin)

def evaluate_f1(gt_dict, pred_dict, rho=0.15):
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

    start_p, end_p = pred_interval
    start_g, end_g = gt_interval


    overlap_start = max(start_p, start_g)
    overlap_end = min(end_p, end_g)
    overlap = max(0, overlap_end - overlap_start)


    union = (end_p - start_p) + (end_g - start_g) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def compute_recall_at_1(ground_truths, predictions, iou_threshold=0.5):
    hits = 0
    total = len(ground_truths)

    for i, gt_interval in enumerate(ground_truths):

        preds = sorted(predictions[i], key=lambda x: x[1], reverse=True)

        best_pred, best_score = preds[0]

        if iou_1d(best_pred, gt_interval) >= iou_threshold:
            hits += 1

    return hits / total



def compute_map_at_iou(ground_truths, predictions, iou_threshold):


    score_label_pairs = []

    for i, gt_interval in enumerate(ground_truths):

        for (pred_interval, score) in predictions[i]:
            iou_val = iou_1d(pred_interval, gt_interval)
            is_positive = (iou_val >= iou_threshold)
            score_label_pairs.append((score, is_positive))


    score_label_pairs.sort(key=lambda x: x[0], reverse=True)


    y_true = [1 if x[1] else 0 for x in score_label_pairs]
    y_scores = [x[0] for x in score_label_pairs]

    mAP = average_precision_score(y_true, y_scores)

    return mAP

def compute_average_map(ground_truths, predictions, iou_thresholds):

    maps = []
    for thresh in iou_thresholds:
        maps.append(compute_map_at_iou(ground_truths, predictions, thresh))
    return np.mean(maps)


def compute_iou_50_and_75(y_truth, y_pred):
    mAP_50 = compute_map_at_iou(y_truth, y_pred, iou_threshold=0.5)
    mAP_75 = compute_map_at_iou(y_truth, y_pred, iou_threshold=0.75)
    average_map = compute_average_map(y_truth, y_pred,
                                    iou_thresholds=np.arange(0.5, 1.0, 0.05))
    print(f"mAP at IoU=0.5: {mAP_50:.4f}")
    print(f"mAP at IoU=0.75: {mAP_75:.4f}")
    print(f"Average mAP over 0.5:0.05:0.95 => {average_map:.4f}")


def compute_ap(gt_binary, sorted_indices, k=5):

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
        ap += (rec - rec_prev) * (prec + prec_prev) / 2.0
        rec_prev = rec
        prec_prev = prec
    return ap
