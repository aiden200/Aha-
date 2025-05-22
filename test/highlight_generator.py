
import os
import json
import cv2
from test.inference import *
import numpy as np

def knapsack_selection(frames_with_index, max_duration, weight, alpha, beta, epsilon):
    """
    Runs the 0/1 knapsack selection on frames using a specific score.
    Each frame has a cost of 1 and a value equal to: (frame[score_key] * weight)
    """
    n = len(frames_with_index)
    # Build DP table: dp[i][j] is max total value using first i frames with budget j.
    dp = [[0 for _ in range(max_duration + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        value = frames_with_index[i - 1]["informative_score"] * alpha \
        + frames_with_index[i - 1]["relevance_score"] * beta \
        + frames_with_index[i - 1]["uncertainty_score"] * epsilon 

        cost = 1  # each frame has cost 1
        for j in range(max_duration + 1):
            if cost <= j:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cost] + value)
            else:
                dp[i][j] = dp[i - 1][j]

    selected_frames = []
    remaining_capacity = max_duration
    for i in range(n, 0, -1):
        if dp[i][remaining_capacity] != dp[i - 1][remaining_capacity]:
            selected_frames.append(frames_with_index[i - 1])
            remaining_capacity -= 1
    selected_frames.reverse()
    
    selected_set = set(frame["idx"] for frame in selected_frames)
    return selected_set

def knapsack_dual_highlight(
        prediction, 
        ground_truth_frames, 
        max_duration,  
        video_path, 
        output_filepath):
    """
    Uses dual knapsack selection on the given prediction:
      - One knapsack uses the relevance_score weighted by 0.65.
      - Another uses the informative_score weighted by 0.65.
    Their selected frame indices are merged (union), and then each selected frame
    is expanded to a window (with the selected frame in the center) to form the highlight.
    Finally, a highlight video is written using OpenCV.
    
    """
    frames = prediction['debug_data']
    n = len(frames)
    if max_duration >= n:
        raise ValueError(f"max_duration ({max_duration}) must be smaller than number of frames ({n})")
    
    # Add index to each frame for backtracking.
    frames_with_index = [{"idx": i, **frame} for i, frame in enumerate(frames)]
    
    with open("outputs/grid_search_params.json", "r") as f:
            best_params = json.load(f)
        
    alpha = best_params["tvsum"]["alpha"]
    beta = best_params["tvsum"]["beta"]
    epsilon = best_params["tvsum"]["epsilon"]
    selected = knapsack_selection(frames_with_index, max_duration, "relevance_score", alpha, beta, epsilon)
    
    
    # 4. Define highlight segments.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    last_frame = ground_truth_frames[-1]
    highlight_indices = set()
    half_width = int(fps // 2) # center the frame
    for idx in selected:
        ground_truth_frame = ground_truth_frames[idx]
        start_idx = max(0, ground_truth_frame - half_width)
        end_idx = min(last_frame + 1, ground_truth_frame + half_width + 1)
        for i in range(start_idx, end_idx):
            highlight_indices.add(i)
    highlight_indices = sorted(list(highlight_indices))
    
    # 5. Write the highlight video.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
    
    frame_idx = 0
    highlight_set = set(highlight_indices)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in highlight_set:
            out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    
    return selected, highlight_indices



def run_model(video_path, query, output_filepath = "highlight_video.mp4"):

    system_prompt="A multimodal AI assistant is helping users with some activities. \
        Below is their conversation, interleaved with the list of video frames received by the assistant."


    args = parse_args('test')
    infer = LiveInferForBenchmark(args)
    max_num_frames = None
    video_frames, fps, video_duration, true_frames_list = load_video_for_testing(video_path, return_true_frames=True, max_num_frames=max_num_frames)    
    conversation = list()
    conversation.append({"role": "system", "content": system_prompt})
    conversation.append({'role': 'user', 'content': query, 'time': 0})
    infer.reset()
    infer.set_fps(fps=fps)
    infer.input_video_stream(video_frames)
    infer.input_query_stream(conversation)
    model_response_list = infer.inference()
    
    
    max_duration = 20
    # frame_selection_width = 30    
    
    merged_selected, highlight_indices = knapsack_dual_highlight(
        {"debug_data": round_numbers(infer.debug_data_list, 3)},
        true_frames_list,
        max_duration,
        video_path,
        output_filepath
    )

if __name__ == "__main__":
    output_filepath = "matrix.mp4"
    run_model(video_path, "Will A Cat Eat Dog Food?", output_filepath)
    
