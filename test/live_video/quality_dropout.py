import cv2 
import random, json, os, csv, math
import numpy as np
import torch

def dropout_simultion(frame, w, h, dropout_type="quality"):
    if dropout_type == "quality":
        degraded = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        frame = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_NEAREST)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    elif dropout_type == "block_noise":
        block_size = 32
        universal_noise_block = np.random.randint(0, 50, (block_size, block_size, 3), dtype=np.uint8)
        for y in range(0, frame.shape[0], block_size):
            for x in range(0, frame.shape[1], block_size):
                if np.random.rand() < 0.1:  # 10% of blocks corrupted
                    actual_block_height = min(block_size, frame.shape[0] - y)
                    actual_block_width = min(block_size, frame.shape[1] - x)
                    noise_to_apply = universal_noise_block[0:actual_block_height, 0:actual_block_width]
                    frame[y : y + actual_block_height, x : x + actual_block_width] = noise_to_apply
    elif dropout_type=="color_banding":
        frame = (frame // 64) * 64
    elif dropout_type == "blackout":
        frame[:] = 0
    
    return frame
    


def get_dropout_times(video_path, dropout_percentage=0.2):
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / input_fps

    dropout_times = []

    current_dropout_duration = 0
    max_dropout_duration = video_duration * dropout_percentage
    while current_dropout_duration < max_dropout_duration:
        dropout_timestamp = random.randint(0, video_duration)
        w = random.randint(3, 6)
        s = max(0, dropout_timestamp - w)
        e = min(video_duration, dropout_timestamp + w)
        dropout_times.append([s, e])
        max_dropout_duration += e - s
    
    return dropout_times







def load_video_for_testing_with_dropout(
    video_file,
    output_fps=1,
    return_true_frames=False,
    max_num_frames=None,
    dropout_types=["quality", "block_noise", "color_banding", "blackout"]
):
    output_resolution = 384
    pad_color = (0, 0, 0)
    cap = cv2.VideoCapture(video_file)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / input_fps
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = output_height = output_resolution

    output_fps = output_fps if output_fps > 0 else max_num_frames / video_duration
    num_frames_total = math.ceil(video_duration * output_fps)
    frame_sec = [i / output_fps for i in range(num_frames_total)]

    dropout_intervals = get_dropout_segments_with_types(num_frames_total, input_fps, possible_dropout_types=dropout_types)
    
    frame_list, cur_time, frame_index = [], 0, 0
    true_frame_index = 0
    true_frame_index_list = []

    def is_dropout(t):
        if dropout_intervals:
            return dropout_intervals[0][0] <= t <= dropout_intervals[0][1]
        return False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
            if is_dropout(frame_index):
                # print(frame_index, dropout_intervals[0][1])
                frame = dropout_simultion(frame, input_width, input_height, dropout_type=dropout_intervals[0][2])
                if frame_index == dropout_intervals[0][1]:
                    dropout_intervals.pop(0)
            # else:
                # print(f"no dropout: {frame_index}")

            # Resize keeping aspect ratio
            if input_width > input_height:
                new_width = output_resolution
                new_height = int((input_height / input_width) * output_resolution)
            else:
                new_height = output_resolution
                new_width = int((input_width / input_height) * output_resolution)

            resized_frame = cv2.resize(frame, (new_width, new_height))
            canvas = cv2.copyMakeBorder(
                resized_frame,
                top=(output_height - new_height) // 2,
                bottom=(output_height - new_height + 1) // 2,
                left=(output_width - new_width) // 2,
                right=(output_width - new_width + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_color,
            )

            if return_true_frames:
                true_frame_index_list.append(true_frame_index)

            # Format to CHW RGB
            frame_list.append(np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)))
            frame_index += 1

        if max_num_frames and len(frame_list) >= max_num_frames:
            break

        cur_time += 1 / input_fps
        true_frame_index += 1

    cap.release()


    if not frame_list:
        return None, None, None, None

    if return_true_frames:
        return torch.tensor(np.stack(frame_list)), output_fps, video_duration, true_frame_index_list

    return torch.tensor(np.stack(frame_list)), output_fps, video_duration



import torch
import numpy as np
import cv2
from pathlib import Path

def save_tensor_as_video(tensor, output_path, fps=1):
    """
    Saves a tensor of shape (T, 3, H, W) as a video file.
    """
    assert tensor.ndim == 4 and tensor.shape[1] == 3, "Expected tensor shape (T, 3, H, W)"
    T, C, H, W = tensor.shape
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for i in range(T):
        frame = tensor[i].numpy()  # shape (3, H, W)
        frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB -> BGR
        writer.write(frame.astype(np.uint8))

    writer.release()
    print(f"Saved video to {output_path}")




def get_dropout_segments_with_types(video_duration_secs, input_fps, 
                                    min_dropout_percentage=0.05, max_dropout_percentage=0.20,
                                    min_segment_duration_secs=1, max_segment_duration_secs=5, # Shorter segments
                                    possible_dropout_types=None, seed=22):
    
    random.seed(seed)
    if possible_dropout_types is None:
        possible_dropout_types = ["quality", "block_noise", "color_banding", "blackout"]
    
    if video_duration_secs == 0 or input_fps == 0:
        return []

    target_total_dropout_secs_float = video_duration_secs * random.uniform(min_dropout_percentage, max_dropout_percentage)

    
    segments = []  #[start_sec_int, end_sec_int, dropout_type]
    accumulated_dropout_secs_int = 0 # Accumulates integer lengths of added segments


    
    min_duration_for_attempts = max(1.0, float(min_segment_duration_secs))
    max_attempts = int(video_duration_secs / min_duration_for_attempts) * 2
    if max_attempts == 0:
        max_attempts = 10 

    for _ in range(max_attempts):
        if accumulated_dropout_secs_int >= target_total_dropout_secs_float:
            break

        # 1. Determine segment length (integer)
        segment_len_candidate_float = random.uniform(min_segment_duration_secs, max_segment_duration_secs)
        segment_len_int = int(round(segment_len_candidate_float))
        if segment_len_int < 1 :
            segment_len_int = 1


        # 2. Adjust segment length for overshoot based on target
        if accumulated_dropout_secs_int + segment_len_int > target_total_dropout_secs_float * 1.2:
            remaining_target_secs_float = target_total_dropout_secs_float - accumulated_dropout_secs_int
            segment_len_int = int(round(max(0.0, remaining_target_secs_float))) # This can round to 0

        # 3. Basic validity for segment_len_int (must be at least 1 second long)
        if segment_len_int < 1:
            continue

        # 4. Determine start time (integer)
        max_allowed_start_time_float = video_duration_secs - segment_len_int
        
        start_time_int = 0

        if max_allowed_start_time_float < 0:
            # This means video_duration_secs < segment_len_int. Video is too short for this segment.
            if not segments: 
                max_fittable_length_in_video = int(math.floor(video_duration_secs))

                if max_fittable_length_in_video < 1: # Video is too short for any segment of at least 1s.
                    continue
                segment_len_int = min(segment_len_int, max_fittable_length_in_video)
                
                if segment_len_int < 1: 
                    continue
                
                start_time_int = 0 
            else: 
                break # Stop trying to add more segments
        else:

            start_time_candidate_float = random.uniform(0, max_allowed_start_time_float)
            start_time_int = int(round(start_time_candidate_float))

        # 5. Calculate end time (integer) and ensure segment fits within video duration
        end_time_int = start_time_int + segment_len_int
        

        if end_time_int > video_duration_secs:
            # This can happen if start_time_int was rounded up close to max_allowed_start_time_float.
            end_time_int = int(math.floor(video_duration_secs))
            new_segment_len_int = end_time_int - start_time_int
            
            if new_segment_len_int < 1: 
                continue
            segment_len_int = new_segment_len_int

        if start_time_int < 0: 
            continue
        if segment_len_int < 1: 
            continue
        if start_time_int >= video_duration_secs and video_duration_secs > 0 : 
             continue
        if start_time_int >= int(math.floor(video_duration_secs)) and int(math.floor(video_duration_secs)) == start_time_int and segment_len_int > 0 : 
             if start_time_int + segment_len_int > int(math.floor(video_duration_secs)): 
                continue


        # 6. Overlap Check (using integer times)
        is_overlapping = False
        for s_existing, e_existing, _ in segments: 
            if max(start_time_int, s_existing) < min(end_time_int, e_existing):
                is_overlapping = True
                break
        if is_overlapping:
            continue

        # 7. Add to segments and update accumulated_dropout_secs
        chosen_type = random.choice(possible_dropout_types)
        segments.append([start_time_int, end_time_int, chosen_type])
        accumulated_dropout_secs_int += segment_len_int # Add the actual integer length

    segments.sort(key=lambda x: x[0])
    return segments



if __name__ == "__main__":

    input_dir = "datasets/tvsum/ydata-tvsum50-v1_1/video"
    caption_metadata_file = "datasets/tvsum/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv"
    video_metadata_file = "datasets/tvsum/videos_metadata.json"

    MAX_CONSECUTIVE_BLACKOUT_FRAMES = 5 # e.g., 5 frames at 1fps = 5 seconds
    input_fps = 1
    MAX_CONSECUTIVE_BLACKOUT_FRAMES = int(5 * input_fps) 
    with open(video_metadata_file, 'r') as f:
        data = json.load(f)
        
    captions = {}
    with open(caption_metadata_file, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            vid_category_code, vid_id, caption, url, length = row
            captions[vid_id] = {
                "query": caption,
            }

    first_video = next(iter(data))
    video_path = os.path.join(input_dir, first_video)

    cap = cv2.VideoCapture(video_path)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_secs = frame_count / input_fps if input_fps > 0 else 0
    num_frames_total = math.ceil(video_duration_secs)
    # print(frame_count, video_duration_secs)
    # assert frame_count == int(video_duration_secs)

    dropout_segments_with_types = get_dropout_segments_with_types(video_duration_secs, input_fps)

    for d in dropout_segments_with_types:
        print(d)
    
    total_dropout_duration = sum([d[1]-d[0] for d in dropout_segments_with_types])
    print(f"Total dropout duration: {total_dropout_duration} out of {num_frames_total}. Approximately {total_dropout_duration/num_frames_total*100}% of the video.")
    # print(dropout_segments_with_types)
    video_frames, fps, video_duration, true_frames_list =  load_video_for_testing_with_dropout(video_path, output_fps=1, return_true_frames=True)
    save_tensor_as_video(video_frames, "test.mp4", fps=1)
    exit(0)
    segment_iter = iter(dropout_segments_with_types)
    current_dropout_segment = next(segment_iter, None)

    consecutive_blackout_frame_count = 0

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time_secs = frame_idx / input_fps
        h, w = frame.shape[:2]
        original_frame_for_dropout = frame.copy() # Keep original if needed later

        # Check if we've passed the current dropout segment
        if current_dropout_segment and current_time_secs >= current_dropout_segment[1]:
            current_dropout_segment = next(segment_iter, None)
            # Reset blackout count when moving out of any dropout segment OR if a segment ends
            # This reset might be too aggressive if segments are back-to-back and both blackout
            # A more precise counter reset would be if the *applied* type isn't blackout.

        apply_this_dropout_type = "none"
        if current_dropout_segment and current_dropout_segment[0] <= current_time_secs < current_dropout_segment[1]:
            apply_this_dropout_type = current_dropout_segment[2]

        # --- Apply blackout limiting logic ---
        if apply_this_dropout_type == "blackout":
            if consecutive_blackout_frame_count < MAX_CONSECUTIVE_BLACKOUT_FRAMES:
                processed_frame = dropout_simulation(original_frame_for_dropout, w, h, "blackout")
                consecutive_blackout_frame_count += 1
            else:
                # Max consecutive blackouts reached. Apply a milder dropout or none.
                mild_dropout_options = ["quality", "color_banding", "none"] 
                chosen_mild_type = random.choice(mild_dropout_options)
                processed_frame = dropout_simulation(original_frame_for_dropout, w, h, chosen_mild_type)
                consecutive_blackout_frame_count = 0 # Reset because this frame wasn't a full blackout
        else:
            processed_frame = dropout_simulation(original_frame_for_dropout, w, h, apply_this_dropout_type)
            if apply_this_dropout_type != "blackout": # Also reset if not a blackout by design
                consecutive_blackout_frame_count = 0
                
        # `processed_frame` is now the frame to use for training
        # ... your model training step ...

    cap.release()