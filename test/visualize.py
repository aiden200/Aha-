import argparse, json
from .tvsum.tvsum_utils import *
from .hisum.hisum_eval import *
import matplotlib.pyplot as plt
import h5py


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=False)
    parser.add_argument('--gold_file', type=str, required=False)
    parser.add_argument('--max_show', type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--hisum_pred_file", type=str, required=False, default="")
    parser.add_argument("--hisum_gold_file", type=str, required=False)
    parser.add_argument("--tvsum_pred_file", type=str, required=False)
    parser.add_argument("--tvsum_gold_file", type=str, required=False)

    args = parser.parse_args()
    print(args)

    
    if args.dataset == "visualize_sota_scores":
        with open(args.hisum_pred_file, "r") as f:
            predictions = json.load(f)
        with h5py.File(args.hisum_gold_file, "r") as hdf:
            
            category_scores = {}
            final_results = list()
            gt_dict = {}
            pred_dict = {}
                
            for prediction in predictions:
                video_uuid = prediction["video_uuid"]
                h5_video_identifier = prediction["h5_identifier"]
                true_frames_list = prediction['true_frames_list']
                vid_ground_truth = list(hdf[h5_video_identifier]["gtscore"])
                if not prediction['debug_data'] or not vid_ground_truth:
                    continue
                    
                # print(len(vid_ground_truth), len(prediction["debug_data"]))
                # print(vid_ground_truth, prediction["debug_data"])
                
                categories = prediction['categories']
                ground_truth_frame_scores = []
                pred_scores = list()
                for i in range(1, min(len(prediction['debug_data']), len(vid_ground_truth))):
                    e = prediction['debug_data'][i]
                    # pred_scores.append(e['relevance_score'])
                    pred_scores.append(
                        args.alpha *e["informative_score"]\
                            + args.beta * e['relevance_score'] \
                                + args.epsilon * e["uncertainty_score"])
                    ground_truth_frame_scores.append(vid_ground_truth[i-1])
                
                pred_scores = np.array(pred_scores)
                ground_truth_frame_scores = np.array(ground_truth_frame_scores)
                for category_name in categories:
                    if category_name not in category_scores:
                        category_scores[category_name] = {"gt_dict": {}, "pred_dict": {}}
                    category_scores[category_name]["gt_dict"][video_uuid] = ground_truth_frame_scores
                    category_scores[category_name]["pred_dict"][video_uuid] = pred_scores
                
                # pred_scores = (pred_scores - np.min(pred_scores)) / (np.max(pred_scores) - np.min(pred_scores))
                pred_scores = np.convolve(pred_scores, np.ones(5)/5, mode='same')
                pred_dict[video_uuid] = pred_scores
                gt_dict[video_uuid] = ground_truth_frame_scores
            

            results = hisum_evaluate_scores(gt_dict, pred_dict)
            hisum_mAP50 = round(float(results["mAP@50"]) * 100, 2)
            hisum_mAP15 = round(float(results["mAP@15"]) * 100, 2)
            hisum_f1 = round(float(results["f1"]) * 100, 2)
            
            metrics = ("mAP@50", "mAP@15", "F1")
            models = {
                "Ours" : (hisum_mAP50, hisum_mAP15, hisum_f1),
                "PGL-SUM": (61.6, 27.45, 55.89),
                "VASNet": (58.69, 25.28, 55.26),
                "SL-module": (58.63, 24.95, 55.31),
                "DSNet": (57.32, 24.35, 50.78),
                # "iPTNet": (55.53, 22.74, 50.5)
            }
            
            x = np.arange(len(metrics))
            width = 0.17
            multiplier = 0
            
            fig, ax = plt.subplots(layout='constrained')
            for model_name, model_result in models.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, model_result, width, label=model_name)
                ax.bar_label(rects, padding=3)
                multiplier += 1
            
            ax.set_ylabel('Metric')
            ax.set_title('Mr.HiSum results')
            ax.set_xticks(x + width, metrics)
            ax.legend(loc='upper left', ncols=3)
            ax.set_ylim(0, 100)

            plt.show()
            plt.savefig("hisum_results_comparison.png")
        



    if args.dataset == "tvsum":
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)
            ground_truths = get_annos(args.gold_file)
            category_scores = {}

            final_results = list()
            gt_dict = {}
            pred_dict = {}

            show_count = 1
                
            for prediction in predictions[5:]:
                if show_count > args.max_show:
                    break
                video_uuid = prediction["video_uuid"]
                true_frames_list = prediction['true_frames_list']
                vid_ground_truth = ground_truths[video_uuid]["importance_scores"] #counterintuitative but this is relevance score
                category_code = ground_truths[video_uuid]["category_code"]
                ground_truth_frame_scores = []
                pred_scores = list()
                uncertainty_scores = list()
                importance_scores = list()
                for i in range(len(prediction['debug_data'])):
                    e = prediction['debug_data'][i]
                    true_frame = true_frames_list[i]
                    # video_times.append(e['video_time'])
                    uncertainty_scores.append(e["uncertainty_score"])
                    importance_scores.append(e['informative_score'])
                    # pred_scores.append(e['relevance_score'])
                    pred_scores.append(
                        args.alpha *e["informative_score"]\
                            + args.beta * e['relevance_score'] \
                                + args.epsilon * e["uncertainty_score"])
                    ground_truth_frame_scores.append(vid_ground_truth[true_frame])
                    # print(e['relevance_score'], vid_ground_truth[true_frame])
                

                pred_scores = np.array(pred_scores)
                # norm_pred = (pred_scores - np.min(pred_scores)) / (np.max(pred_scores) - np.min(pred_scores))
                pred_scores = np.convolve(pred_scores, np.ones(5)/5, mode='same')
                # pred_scores = list(norm_pred)

                # importance_scores = np.array(importance_scores)
                # importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores))
                # importance_scores = list(importance_scores)
                
                x = list(range(len(pred_scores)))
                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1])
                # Chart 1: Predicted vs. Ground Truth Relevance
                axes[0].plot(x, pred_scores, label='Predicted Relevance', color='tab:blue', linewidth=2)
                axes[0].plot(x, importance_scores, label='Predicted Importance', color='tab:red', linewidth=2)
                axes[0].plot(x, ground_truth_frame_scores, label='Ground Truth Relevance', color='tab:green', linestyle='--', linewidth=2)
                axes[0].set_ylabel('Score')
                axes[0].legend()
                axes[0].set_title('Predicted vs. Ground Truth Relevance Over Time')

                # Chart 2: Uncertainty
                axes[1].plot(x, uncertainty_scores, label='Uncertainty', color='tab:red', linewidth=2)
                axes[1].set_ylabel('Uncertainty')
                axes[1].set_xlabel('Frame')
                axes[1].set_title('Uncertainty Score Over Time')

                plt.tight_layout()
                plt.show()
                plt.savefig(f"results_{show_count}.png")

                show_count += 1
                
    if args.dataset == "hisum":
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)
        with h5py.File(args.gold_file, "r") as hdf:
            
            category_scores = {}
            final_results = list()
            gt_dict = {}
            pred_dict = {}

            show_count = 1
                
            for prediction in predictions[5:]:
                if show_count > args.max_show:
                    break
                video_uuid = prediction["video_uuid"]
                h5_video_identifier = prediction["h5_identifier"]
                true_frames_list = prediction['true_frames_list']
                vid_ground_truth = list(hdf[h5_video_identifier]["gtscore"])
                if not prediction['debug_data'] or not vid_ground_truth:
                    continue
                    
                # print(len(vid_ground_truth), len(prediction["debug_data"]))
                # print(vid_ground_truth, prediction["debug_data"])
                
                categories = prediction['categories']
                ground_truth_frame_scores = []
                pred_scores = list()
                uncertainty_scores = list()
                importance_scores = list()
                for i in range(1, min(len(prediction['debug_data']), len(vid_ground_truth))):
                    e = prediction['debug_data'][i]
                    # pred_scores.append(e['relevance_score'])
                    pred_scores.append(
                        args.alpha *e["informative_score"]\
                            + args.beta * e['relevance_score'] \
                                + args.epsilon * e["uncertainty_score"])
                    ground_truth_frame_scores.append(vid_ground_truth[i-1])
                    uncertainty_scores.append(e["uncertainty_score"])
                    importance_scores.append(e['informative_score'])
                    
                pred_scores = np.array(pred_scores)
                ground_truth_frame_scores = np.array(ground_truth_frame_scores)
                for category_name in categories:
                    if category_name not in category_scores:
                        category_scores[category_name] = {"gt_dict": {}, "pred_dict": {}}
                    category_scores[category_name]["gt_dict"][video_uuid] = ground_truth_frame_scores
                    category_scores[category_name]["pred_dict"][video_uuid] = pred_scores
                
                # pred_scores = (pred_scores - np.min(pred_scores)) / (np.max(pred_scores) - np.min(pred_scores))
                pred_scores = np.convolve(pred_scores, np.ones(5)/5, mode='same')
                pred_dict[video_uuid] = pred_scores
                gt_dict[video_uuid] = ground_truth_frame_scores
                
                
                x = list(range(len(pred_scores)))
                fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1])
                # Chart 1: Predicted vs. Ground Truth Relevance
                axes[0].plot(x, pred_scores, label='Predicted Relevance', color='tab:blue', linewidth=2)
                axes[0].plot(x, importance_scores, label='Predicted Importance', color='tab:red', linewidth=2)
                axes[0].plot(x, ground_truth_frame_scores, label='Ground Truth Relevance', color='tab:green', linestyle='--', linewidth=2)
                axes[0].set_ylabel('Score')
                axes[0].legend()
                axes[0].set_title('Predicted vs. Ground Truth Relevance Over Time')

                # Chart 2: Uncertainty
                axes[1].plot(x, uncertainty_scores, label='Uncertainty', color='tab:red', linewidth=2)
                axes[1].set_ylabel('Uncertainty')
                axes[1].set_xlabel('Frame')
                axes[1].set_title('Uncertainty Score Over Time')

                plt.tight_layout()
                plt.show()
                plt.savefig(f"hisum_results_{show_count}.png")

                show_count += 1



            