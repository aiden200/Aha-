import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from scipy.stats import pearsonr, spearmanr

def segment_into_shots(scores, shot_length=1, fps=1):
    n_frames = len(scores)
    seg_size = shot_length * fps
    segments = [scores[i:i+seg_size] for i in range(0, n_frames, seg_size)]
    segment_scores = np.array([np.mean(s) for s in segments])
    return segment_scores, len(segments), seg_size

def get_top_k_indices(scores, k):
    return np.argsort(scores)[-k:]

def hisum_mean_average_precision(gt_dict, pred_dict, rho=0.5):
    ap_list = []
    for video_id in gt_dict:
        gt_scores, pred_scores = gt_dict[video_id], pred_dict[video_id]
        pred_seg, n_seg, seg_size = segment_into_shots(pred_scores)
        gt_seg, _, _ = segment_into_shots(gt_scores)

        k = max(1, int(rho * n_seg))
        gt_labels = np.zeros(n_seg)
        gt_top_indices = get_top_k_indices(gt_seg, k)
        gt_labels[gt_top_indices] = 1

        ap = average_precision_score(gt_labels, pred_seg)
        if not np.isnan(ap):
            ap_list.append(ap)

    return np.mean(ap_list)

def hisum_f1_score_summarization(gt_dict, pred_dict, budget=0.15, shot_length=1):
    f1_list = []
    for video_id in gt_dict:
        gt_scores = gt_dict[video_id]
        pred_scores = pred_dict[video_id]
        n_frames = len(gt_scores)

        # Uniform segmentation into fixed-length shots
        seg_size = shot_length  # since fps=1, this is just 5 frames
        boundaries = [(i, min(i + seg_size, n_frames)) for i in range(0, n_frames, seg_size)]

        # Compute per-shot predicted scores
        shot_scores = [np.mean(pred_scores[start:end]) for start, end in boundaries]

        # Knapsack-style selection: pick top-scoring shots until budget is filled
        total_budget = int(budget * n_frames)
        selected = np.zeros(n_frames, dtype=bool)
        acc = 0
        for idx in np.argsort(shot_scores)[::-1]:
            start, end = boundaries[idx]
            length = end - start
            if acc + length <= total_budget:
                selected[start:end] = True
                acc += length
            if acc >= total_budget:
                break

        # Ground truth: top budget-percent of frames
        gt_selected = gt_scores >= np.percentile(gt_scores, 100 * (1 - budget))
        pred_selected = selected
        

        f1 = f1_score(gt_selected, pred_selected)
        # if f1 == 0:
        #     print(pred_scores, gt_scores)
        #     print(gt_selected,pred_selected)
        f1_list.append(round(f1, 2))
    # print(f1_list)

    return np.mean(f1_list)


def hisum_evaluate_scores(gt_dict, pred_dict, print_logs=True):
    mse_list = []
    mae_list = []
    pearson_list = []
    spearman_list = []

    # for video_id in gt_dict:
    #     gt = gt_dict[video_id]
    #     pred = pred_dict[video_id]

    #     if len(gt) != len(pred):
    #         continue  # sanity check

    #     mse_list.append(mean_squared_error(gt, pred))
    #     mae_list.append(mean_absolute_error(gt, pred))

    #     if len(gt) > 1:  # correlation requires more than 1 point
    #         pearson_corr, _ = pearsonr(gt, pred)
    #         spearman_corr, _ = spearmanr(gt, pred)
    #     else:
    #         pearson_corr = spearman_corr = 0.0
        
    #     pearson_list.append(pearson_corr)
    #     spearman_list.append(spearman_corr)
    
    map50 = hisum_mean_average_precision(gt_dict, pred_dict, rho=.5)
    map15 = hisum_mean_average_precision(gt_dict, pred_dict, rho=.15)
    f1 = hisum_f1_score_summarization(gt_dict, pred_dict)

    if print_logs:
        print("Overall Evaluation:")
        # print(f"  MSE: {np.mean(mse_list):.4f}")
        # print(f"  MAE: {np.mean(mae_list):.4f}")
        # print(f"  Pearson: {np.mean(pearson_list):.4f}")
        # print(f"  Spearman: {np.mean(spearman_list):.4f}")
        print(f"  mAP@50: {map50:.4f}")
        print(f"  mAP@15: {map15:.4f}")
        print(f"  F1: {f1:.4f}")
        

    return {
        # "mse": np.mean(mse_list),
        # "mae": np.mean(mae_list),
        # "pearson": np.mean(pearson_list),
        # "spearman": np.mean(spearman_list),
        "mAP@50": map50,
        "mAP@15": map15,
        "f1": f1
    }
