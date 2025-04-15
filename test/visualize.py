import argparse, json
from .tvsum.tvsum_utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--max_show', type=int, default=1)

    args = parser.parse_args()
    print(args)




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
                    pred_scores.append(e['relevance_score'])
                    importance_scores.append(e['informative_score'])
                    ground_truth_frame_scores.append(vid_ground_truth[true_frame])
                    # print(e['relevance_score'], vid_ground_truth[true_frame])
                

                pred_scores = np.array(pred_scores)
                norm_pred = (pred_scores - np.min(pred_scores)) / (np.max(pred_scores) - np.min(pred_scores))
                pred_scores = list(norm_pred)

                importance_scores = np.array(importance_scores)
                importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores))
                importance_scores = list(importance_scores)
                
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

                show_count += 1


            