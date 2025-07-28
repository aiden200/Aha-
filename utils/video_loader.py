
import os, cv2, json
import multiprocessing as mp
import tqdm

def is_valid_video_cap(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    if not ret or frame is None:
        return False
    cap.release()
    return True



def get_all_files(directory):
    relative_file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file[-4:] == ".mp4":
                # Get the relative path by removing the directory part from the absolute path
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                relative_file_list.append(relative_path)
    print(f"Loading {len(relative_file_list)} files")
    return relative_file_list

def get_video_duration_and_fps(args):
    try:
        file, video_root = args
        path = os.path.join(video_root, file)
        valid_vid = is_valid_video_cap(path)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        return file, {'duration': duration, 'fps': fps, 'path': path, 'frame_count': frame_count, 'valid': valid_vid}
    except Exception as e:
        print(f"Corrupted file {file}")
        return None, None


def check_metadata(metadata_path, video_root):
    if os.path.exists(metadata_path):
        return
    else:
        metadata = {}
        files = get_all_files(video_root)
        with mp.Pool(20) as pool:
            results = list(tqdm.tqdm(pool.imap(
                get_video_duration_and_fps, [(file, video_root) for file in files]),
                total=len(files), desc=f'prepare {metadata_path}...'))
        for key, value in results:
            if key and value:
                metadata[key] = value
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

