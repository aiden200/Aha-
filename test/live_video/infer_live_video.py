
import collections, math, json, copy, random, os, csv, sys, yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
from test.arl_scout.prepare_data import generate_plot


ARL_TICKS = [
    (0, 0, "TV"),
    (28, 28, "Dark Room"),
    (48, 48, "Pitch Black"),
    (58, 58, "Door"),
    (78, 78, "Move to bright room w/ shovel"),
    (122, 122, "turn to door"),
    (131, 161, "static at door"),
    (166, 166, "rapid turn to TV room"),
    (200, 200, "Moved closer to TV"),
    (202, 245, "static"),
    (245, 266, "Moved closer to TV, static"),
    (266, 289, "static TV"),
    (289, 298, "turn to wall/poster"),
    (305, 305, "full turn to poster"),
    (357, 357, "turn away from poster"),
    (375, 375, "face hallway"),
    (411, 426, "move into dark room"),
    (445, 445, "face dark room window"),
    (471, 471, "move to door"),
    (503, 503, "move in dark room"),
    (529, 529, "turn to lit area"),
    (638, 638, "turn & move to new area"),
    (696, 696, "big move to lit room"),
    (725, 725, "slight turn"),
    (767, 767, "move to calendar"),
    (849, 871, "massive movement"),
    (933, 933, "move to water jug"),
    (955, 955, "turn to cubes"),
    (1000, 1000, "turn to water jug"),
    (1020, 1020, "move to water jug"),
    (1031, 1031, "switch angle"),
]


HUBBLE_SPACE_TELESCOPE_TICKS = [
    (590, 590, "Launch"),
    (747, 747, "Opening bay doors"),
    (1240, 1240, "Taking telescope out of payload bay"),
    (1490, 1490, "Deploying the solar arrays"),
    (1568, 1568, "Deploying the high gain antennas"),
    (1616, 1616, "Unfurling the first solar array"),
    (1816, 1816, "EVA preparation"),
    (1884, 1884, "Unfurling the second solar array"),
    (1920, 1920, "Second solar array gets stuck"),
    (2070, 2070, "Disable tension monitoring software to unfurl the solar array"),
    (2185, 2185, "Go for Hubble release"),
    (2347, 2347, "Student experiment"),
    (2630, 2630, "Commands sent to open aperture door"),
    (2745, 2745, "Thank you to training crew"),
    (2800, 2800, "Thoughts on historical significance"),
    (3009, 3009, "Closing bay doors"),
    (3058, 3058, "Shuttle re-entry and landing"),
    (3299, 3299, "Astronauts exiting Shuttle"),
]

def find_ticks(scores):

    smoothed = savgol_filter(scores, window_length=15, polyorder=3)
    thresh = smoothed.mean() + 0.5*smoothed.std()

    # 4) find peaks that exceed that threshold, and are at least D frames apart  
    fps = 1.0                        # or whatever your sampling rate is  
    min_separation = 10             # seconds between distinct peaks  
    distance = int(min_separation * fps)

    peaks, props = find_peaks(
        smoothed,
        height=thresh,        # only peaks above this absolute value
        prominence=0.02,      # only “sharp” spikes (tune as needed)
        distance=distance
    )

    # 5) convert to times (seconds)
    peak_times = peaks / fps

    print("Detected spikes at:", peak_times)
    return list(peak_times)


def truncate_sig(x, sig=3):
    if x == 0:
        return 0
    return float(f"{x:.{sig}g}")

def round_numbers(data, n):
    if isinstance(data, list):
        return [round_numbers(d, n) for d in data]
    elif isinstance(data, dict):
        return {k: round_numbers(v, n) for k, v in data.items()}
    elif isinstance(data, float):
        if abs(data) <= 10**(-n):
            return truncate_sig(data, n)
        else:
            return round(data, n)
    return data

def infer_on_live_video(infer, query, skip, video_frames, system_prompt, output_file, parent_dir, frame_folder, ticks, fps):
    verbose_output_file = os.path.join(parent_dir, "verbose.json")
    if not skip:
        if video_frames == None:
            raise ValueError("Error, no video frames")
        conversation = list()
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({'role': 'user', 'content': query, 'time': 0})


        infer.reset()
        infer.set_fps(fps=fps)
        infer.input_video_stream(video_frames)
        infer.input_query_stream(conversation)
        model_response_list = infer.inference(verbose=True, total=len(video_frames))
        

        results = round_numbers(infer.debug_data_list, 3)
        with open(output_file, "w") as f:
            json.dump(results, f)

        with open(verbose_output_file, "w") as f:
            json.dump(model_response_list, f)
    else:
        with open(output_file, "r") as f:
            results = json.load(f)
        
        with open(verbose_output_file, "r") as f:
            model_response_list = json.load(f)
    

    model_response_formated = {} 
    for response in model_response_list:
        if response["role"] == "assistant":
            model_response_formated[response["time"]] = response["content"] 
        

    
    times = [d["time"] for d in results]
    informative_scores = [d["informative_score"] for d in results]
    relevance_scores = [d["relevance_score"] for d in results]
    uncertainty_scores = [d["uncertainty_score"] for d in results]
    # relevance_scores = np.convolve(relevance_scores, np.ones(5)/5, mode='same')

    # Create the plot
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(times, informative_scores, label="Informative Score", alpha=0.2)
    ax.plot(times, relevance_scores, label="Relevance Score", color="BLACK")
    ax.plot(times, uncertainty_scores, label="Uncertainty Score", alpha=1.0)

    gt_tics = []
    for idx, (start, end, label) in enumerate(ticks):
        gt_tics.append(start)
        color = f"C{idx % 10}"  # Cycle through matplotlib's default color cycle
        alpha = 0.3
        mid = (start + end) / 2
        if start == end:
            end += 1
            alpha = 0.8
            mid -= 3.5
        ax.axvspan(start, end, ymin=0, ymax=1, color=color, alpha=alpha)
        ax.text(mid, 0, label, rotation=90, va='bottom', ha='center', fontsize=25, color='black', clip_on=True)
    
    automatic_tics = find_ticks(np.array(relevance_scores))
    for t in automatic_tics:
        real_time = t + 60*14 + 38
        min = int(real_time // 60)
        sec = int(real_time % 60)
        print(f"{min}:{sec}")
    

    if 0 not in automatic_tics or 0.0 not in automatic_tics:
        automatic_tics = [0] + automatic_tics
    # print(automatic_tics)
    # print(gt_tics)
    for idx in automatic_tics:
        
        color = "black"
        alpha = 1.0
        ax.axvspan(idx, idx+1, ymin=0, ymax=1, color=color, alpha=alpha)

    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.set_title("Scores over Time with Scene Annotations")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, "visualization.png"))
    plt.show()

    
    # plt.figure(figsize=(14, 7))
    # plt.plot(times, informative_scores, label="Informative Score")
    # plt.plot(times, relevance_scores, label="Relevance Score")
    # plt.plot(times, uncertainty_scores, label="Uncertainty Score")
    # plt.xlabel("Time")
    # plt.ylabel("Score")
    # plt.title("Scores over Time")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(parent_dir, "visualization.png"))

    os.makedirs(os.path.join(parent_dir, "stiched"), exist_ok=True)

    stiched_img_paths = []
    # print(model_response_formated)
    
    for idx in range(len(results)):
        # Load frame
        frame_path = os.path.join(frame_folder, f"frame{idx:03d}.jpg")
        frame_img = Image.open(frame_path).convert("RGB")
        
        agent_response = None
        if idx in model_response_formated:
            agent_response = model_response_formated[idx]

        # Generate plot
        plot_img = generate_plot(idx, results, agent_response)

        # Resize plot to match frame height
        plot_img = plot_img.resize((plot_img.width * frame_img.height // plot_img.height, frame_img.height), resample=Image.LANCZOS)

        # Stitch side-by-side
        stitched_width = frame_img.width + plot_img.width
        stitched_img = Image.new("RGB", (stitched_width, frame_img.height))
        stitched_img.paste(frame_img, (0, 0))
        stitched_img.paste(plot_img, (frame_img.width, 0))

        # Save stitched image
        stiched_img_path = os.path.join(parent_dir, "stiched", f"stitched_{idx}.jpg")
        stiched_img_paths.append(stiched_img_path)
        stitched_img.save(stiched_img_path)
    
    fps = 1

    # Read first frame to get dimensions
    frame = cv2.imread(stiched_img_paths[0])
    height, width, layers = frame.shape

    output_video_path = os.path.join(parent_dir, "arl_scout_results_stiched_video.mp4")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 file
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame
    for stiched_img in stiched_img_paths:
        frame = cv2.imread(stiched_img)
        video.write(frame)

    video.release()
    print(f"Video saved to {output_video_path}")
    
    print(f"Results saved at: {output_file}")






