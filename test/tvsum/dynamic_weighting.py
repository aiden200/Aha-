import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameGatingMLP(nn.Module):
    """
    Given per-frame head outputs [R_t, I_t, U_t], produces a
    softmax over 3 coefficients [alpha_t, beta_t, gamma_t].
    Trained with an MSE loss between the weighted sum and the true y_t.
    """
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, head_scores: torch.Tensor) -> torch.Tensor:
        """
        head_scores: Tensor of shape (batch, 3)
        returns:     Tensor of shape (batch, 3) of weights summing to 1
        """
        h = F.relu(self.fc1(head_scores))
        return F.softmax(self.fc2(h), dim=-1)


class UnsupervisedEMAAdaptor:
    def __init__(self, alpha_var=0.1, alpha_dis=0.1, eps=1e-6):
        self.alpha_var = alpha_var
        self.alpha_dis = alpha_dis
        self.eps       = eps
        self.ema_mean  = {}
        self.ema_var   = {}
        self.ema_ens   = {}
        self.ema_dis   = {}

    def update(self, head_scores: dict):
        # compute ensemble average
        ens = sum(head_scores.values()) / len(head_scores)
        for head, val in head_scores.items():
            # 1) running mean
            m_old = self.ema_mean.get(head, val)
            m_new = self.alpha_var * val + (1 - self.alpha_var) * m_old
            self.ema_mean[head] = m_new

            # 2) variance EMA
            dev2 = (val - m_new)**2
            v_old = self.ema_var.get(head, dev2)
            self.ema_var[head] = self.alpha_var * dev2 + (1 - self.alpha_var) * v_old

            # 3) running ensemble
            e_old = self.ema_ens.get(head, ens)
            self.ema_ens[head] = self.alpha_dis * ens + (1 - self.alpha_dis) * e_old

            # 4) disagreement EMA
            d = abs(val - ens)
            d_old = self.ema_dis.get(head, d)
            self.ema_dis[head] = self.alpha_dis * d + (1 - self.alpha_dis) * d_old

    def get_weights(self, head_names: list):
        # ensure each requested head has been initialized
        for head in head_names:
            self.ema_var.setdefault(head, 0.0)
            self.ema_dis.setdefault(head, 0.0)

        inv_var = {h: 1.0/(self.ema_var[h] + self.eps) for h in head_names}
        inv_dis = {h: 1.0/(self.ema_dis[h] + self.eps) for h in head_names}
        # combine
        score = {h: inv_var[h]*inv_dis[h] for h in head_names}
        s = sum(score.values()) or 1.0
        return {h: score[h]/s for h in head_names}






class EMAWeightedAdaptor:
    """
    Keeps an EMA of *squared* errors for each head,
    and computes weights ∝ 1 / (ema_mse + eps).
    """
    def __init__(self, alpha: float = 0.1, eps: float = 1e-6):
        self.alpha   = alpha
        self.eps     = eps
        # initialize EMA of MSE for each head
        self.ema_mse = {
            'relevance': 1.0,
            'informative': 1.0,
            'uncertainty': 1.0,
            'penalty': 1.0
        }

    def update(self,
               pred_scores: dict,
               true_value: float):
        """
        Update each head's EMA of squared error.
        pred_scores: {'relevance': float, 'informative': float, 'uncertainty': float}
        true_value:  float ground-truth highlight
        """
        for head, pred in pred_scores.items():
            se = (pred - true_value) ** 2
            self.ema_mse[head] = self.alpha * se + (1 - self.alpha) * self.ema_mse[head]

    def get_weights(self,
                    head_names: list = None) -> dict:
        """
        head_names: list of heads you want weights for (e.g. list(preds.keys())).
        If None, uses all keys in ema_mse.
        Returns a dict head -> normalized weight.
        """
        if head_names is None:
            head_names = list(self.ema_mse.keys())
        inv = {h: 1.0 / (self.ema_mse[h] + self.eps) for h in head_names}
        total = sum(inv.values())
        return {h: inv[h] / total for h in head_names}


class StableEMAAdaptor(UnsupervisedEMAAdaptor):
    def __init__(self, alpha_var=0.05, alpha_dis=0.05, eps=1e-6, floor=1e-2):
        super().__init__(alpha_var, alpha_dis, eps)
        self.floor = floor
        self.frame_count = 0

    def update(self, head_scores):
        super().update(head_scores)
        # clamp variance / disagreement to floor
        for head in self.ema_var:
            self.ema_var[head] = max(self.ema_var[head], self.floor)
            self.ema_dis[head] = max(self.ema_dis[head], self.floor)

        # periodic reset every 1000 frames:
        self.frame_count += 1
        if self.frame_count % 1000 == 0:
            # re‑initialize to train‐time values or uniform:
            for head in self.ema_var:
                self.ema_var[head]   = self.floor
                self.ema_dis[head]   = self.floor

    def get_weights(self, head_names=None):
        dyn = super().get_weights(head_names)
        # blend 50/50 with uniform
        N = len(dyn)
        return {h: 0.5*dyn[h] + 0.5*(1/N) for h in dyn}




import json
from tvsum_utils import *
import random
from tqdm import tqdm

def evaluate_dynamic(pred_file, gold_file, model="ema"):
    with open(pred_file, "r") as f:
        predictions = json.load(f)

    ground_truths = get_annos(gold_file)


    indices = list(range(len(predictions)))
    random.shuffle(indices)
    predictions = [predictions[i] for i in indices]
    import math
    train, test = predictions[:math.floor(len(predictions)*.8)], predictions[-int(len(predictions)*.2):]

    predictions = train
    print(len(train), len(test))


    final_results = list()
    gt_dict = {}
    pred_dict = {}
    device = "cuda"

    if model=="ema":
        losses = []
        for _ in range(1):
            gate = EMAWeightedAdaptor(alpha=0.1)
            thr = 0.0
            eps = 0.2
            for prediction in tqdm(predictions):
                video_uuid = prediction["video_uuid"]
                true_frames_list = prediction['true_frames_list']
                vid_ground_truth = ground_truths[video_uuid]["importance_scores"]
                frame_preds = []
                frame_gts = []
                for i in range(len(prediction['debug_data'])):
                    e = prediction['debug_data'][i]
                    true_frame = true_frames_list[i]
                    y_true = vid_ground_truth[true_frame]

                    rel = e["relevance_score"]
                    inf = e["informative_score"]
                    unc = e["uncertainty_score"]
                    diff = max(0.0, unc - thr)
                    pen = diff * eps

                    preds = {
                        'relevance': rel,
                        'informative': inf,
                        'uncertainty': unc,
                        'penalty': pen,
                    }
                    gate.update(preds, vid_ground_truth[true_frames_list[i]])
                    weights = gate.get_weights()
                    final_score = sum(weights[h] * preds[h] for h in preds)

                    frame_preds.append(final_score)
                    frame_gts.append(vid_ground_truth[true_frames_list[i]])
            
            for prediction in tqdm(test):
                video_uuid = prediction["video_uuid"]
                true_frames_list = prediction['true_frames_list']
                vid_ground_truth = ground_truths[video_uuid]["importance_scores"]
                frame_preds = []
                frame_gts = []
                for i in range(len(prediction['debug_data'])):
                    e = prediction['debug_data'][i]
                    true_frame = true_frames_list[i]
                    y_true = vid_ground_truth[true_frame]

                    rel = e["relevance_score"]
                    inf = e["informative_score"]
                    unc = e["uncertainty_score"]
                    diff = max(0.0, unc - thr)
                    pen = diff * eps

                    preds = {
                        'relevance': rel,
                        'informative': inf,
                        'uncertainty': unc,
                        'penalty': pen,
                    }
                    # gate.update(preds, vid_ground_truth[true_frames_list[i]])
                    weights = gate.get_weights()
                    final_score = sum(weights[h] * preds[h] for h in preds)

                    frame_preds.append(final_score)
                    frame_gts.append(vid_ground_truth[true_frames_list[i]])
                pred_dict[video_uuid] = np.array(frame_preds)
                gt_dict[video_uuid] = np.array(frame_gts)

                    # pred_dict[video_uuid] = pred_scores
                    # gt_dict[video_uuid] = ground_truth_frame_scores


            mAP50, mAP15, top5, spearman, kendall = evaluate_tvsum(gt_dict, pred_dict)
            # print(f"EMA‑gate top5 mAP: {top5}")
            losses.append(top5)

        print(f"EMA‑gate top5 mAP: {sum(losses)/len(losses)}")
        print(losses)

    elif model=="uema":

        losses = []
        for _ in range(1):
            gate = StableEMAAdaptor(alpha_var=0.05, alpha_dis=0.1)
            weights = gate.get_weights(head_names=['relevance','informative','uncertainty','penalty'])

            thr = 0.0
            eps = 0.2
            for prediction in tqdm(predictions + test):
                video_uuid = prediction["video_uuid"]
                true_frames_list = prediction['true_frames_list']
                vid_ground_truth = ground_truths[video_uuid]["importance_scores"]
                frame_preds = []
                frame_gts = []
                for i in range(len(prediction['debug_data'])):
                    e = prediction['debug_data'][i]
                    true_frame = true_frames_list[i]
                    y_true = vid_ground_truth[true_frame]

                    rel = e["relevance_score"]
                    inf = e["informative_score"]
                    unc = e["uncertainty_score"]
                    diff = max(0.0, unc - thr)
                    pen = diff * eps

                    preds = {
                        'relevance': rel,
                        'informative': inf,
                        'uncertainty': unc,
                        'penalty': pen,
                    }
                    weights = gate.get_weights(head_names=list(preds.keys()))
                    gate.update(preds)
                    weights = gate.get_weights(head_names=list(preds.keys()))
                    # weights = {h: 0.5*weights[h] + 0.5*(1/len(weights)) for h in weights}

                    final_score = sum(weights[h]*preds[h] for h in preds)

                    frame_preds.append(final_score)
                    frame_gts.append(vid_ground_truth[true_frames_list[i]])
                pred_dict[video_uuid] = np.array(frame_preds)
                gt_dict[video_uuid] = np.array(frame_gts)

            
            mAP50, mAP15, top5, spearman, kendall = evaluate_tvsum(gt_dict, pred_dict)
            # print(f"EMA‑gate top5 mAP: {top5}")
            losses.append(top5)

        print(f"EMA‑gate top5 mAP: {sum(losses)/len(losses)}")
        print(losses)
    else:
        gate = FrameGatingMLP(hidden_dim=8).to(device)
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
        num_epochs = 2


        for prediction in tqdm(predictions):
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                video_uuid = prediction["video_uuid"]
                true_frames_list = prediction['true_frames_list']
                vid_ground_truth = ground_truths[video_uuid]["importance_scores"]

                for i in range(len(prediction['debug_data'])):
                    e = prediction['debug_data'][i]
                    true_frame = true_frames_list[i]
                    y_true = vid_ground_truth[true_frame]

                    
                    head_scores = torch.tensor([
                        e["relevance_score"],
                        e["informative_score"],
                        e["uncertainty_score"]
                    ], dtype=torch.float32, device=device)

                    head_scores = head_scores.unsqueeze(0)

                    weights = gate(head_scores)            # (batch,3)
                    y_pred = (weights * head_scores).sum(dim=1)  # weighted sum
                    
                    y_true = torch.tensor([y_true], dtype=torch.float32, device=device)

                    loss = mse_loss(y_pred, y_true)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    # print(f"Epoch {epoch} MSE: {epoch_loss/len(prediction['debug_data']):.4f}")
        
        gate.eval()
        print("Evaluating")


        for prediction in tqdm(test):
            video_uuid = prediction["video_uuid"]
            true_frames_list = prediction['true_frames_list']
            vid_ground_truth = ground_truths[video_uuid]["importance_scores"] #counterintuitative but this is relevance score
            category_code = ground_truths[video_uuid]["category_code"]
            ground_truth_frame_scores = []
            pred_scores = list()
            with torch.no_grad():
                for i in range(len(prediction['debug_data'])):
                    e = prediction['debug_data'][i]
                    true_frame = true_frames_list[i]

                    head_scores = torch.tensor([
                        e["relevance_score"],
                        e["informative_score"],
                        e["uncertainty_score"]
                    ], dtype=torch.float32, device=device)

                    head_scores = head_scores.unsqueeze(0)

                    weights = gate(head_scores)  
                    curr_pred_score = (weights * head_scores).sum(dim=1).item()


                    pred_scores.append(curr_pred_score)
                    ground_truth_frame_scores.append(vid_ground_truth[true_frame])
                
                pred_scores = np.array(pred_scores)
                # pred_scores = np.convolve(pred_scores, np.ones(5)/5, mode='same')
                # kernel = np.ones(5) / 5
                # relevance_scores = np.convolve(relevance_scores, kernel, mode='full')[:len(relevance_scores)]
            

                ground_truth_frame_scores = np.array(ground_truth_frame_scores)
                

                pred_dict[video_uuid] = pred_scores
                gt_dict[video_uuid] = ground_truth_frame_scores

        mAP50, mAP15, top_5_map, spearman, kendall = evaluate_tvsum(gt_dict, pred_dict)
        print(f"Model: {model}, top5 mAP: {top_5_map}")


pred_file = "outputs/tvsum_dynamic_sink/eval/tvsum_test-random_prompt-pred.json"
gold_file = "datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv"

evaluate_dynamic(pred_file, gold_file, "uema")