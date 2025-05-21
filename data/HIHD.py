import json, os, shutil, csv, time
import cv2
from tqdm import tqdm, trange
import math
import torch
import random
import numpy as np
from typing import DefaultDict
import pandas as pd

from transformers.utils import logging
from .stream import StreamMixIn
from .utils import reformat_example_for_debug, resize_and_pad_frame, dropout_simultion, DictWithTo
logger = logging.get_logger(__name__)





class HIHD(StreamMixIn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        annos, self.annos = self.annos, list()

        for anno in tqdm(annos):
            curr_info = annos[anno]
            video_uid = curr_info['video_uid']
            query = curr_info["query"]
            scores = curr_info["scores"]
            
            if video_uid not in self.metadata:
                logger.warning(f'Missmatch in captions and annotations: {video_uid}')
                continue
            if not self.metadata[video_uid]["valid"]:
                logger.warning(f"Video {video_uid} not loaded, possibly corrupted")
                continue
            duration = self.metadata[video_uid]['duration']
            conversation, current_frame = list(), 0
            conversation.append({'role': 'user', 'content': query, 'learn': False})
            
            # One score per duration
            for i in range(len(scores)):
                conversation.append({'role': 'stream', 'num_frames': 1, 'learn': True, 'related': scores[i]})
            

            final_frame = math.floor(duration * self.frame_fps)
            if final_frame < len(conversation):
                conversation = conversation[:final_frame + 1]
            
            self.annos.append({
                'conversation': conversation,
                'load_ranges': {video_uid: range(0, final_frame)}
            })
        print(f'Dataset {self.__class__.__name__} has {len(self)} examples. Example data: {reformat_example_for_debug(self[0])}')

    def get_annos(self) -> dict:
        s = time.time()
        annotations = {}
        anno_path = os.path.join(self.anno_file)
        assert os.path.exists(anno_path)
        df = pd.read_csv(anno_path)
        df = df[df["training_split"]=="train"]
        df["scores"] = df["scores"].apply(json.loads)
        df["quality_dropout"] = df["quality_dropout"].apply(json.loads)
        all_files = os.listdir(self.video_root)
        self.quality_dropout = {}
        for i in range(len(df)):
            row = df.iloc[i]
            scores = row["scores"]
            quality_dropout = row["quality_dropout"]
            query = row["query"]
            duration = row["duration"]
            youtube_id = row["youtube_id"]
            video_uid = youtube_id + ".mp4"
            if video_uid in all_files:
                annotations[youtube_id] = {
                    "scores": scores,
                    # "quality_dropout": quality_dropout,
                    "query": query,
                    "duration": duration,
                    "video_uid": video_uid
                }
            self.quality_dropout[video_uid] = quality_dropout
        
        print(f"HIHD loaded {len(annotations)} videos out of {len(df)} in {time.time()-s:.2f} seconds")
        logger.info(f"HIHD loaded {len(annotations)} videos out of {len(df)}")

        return annotations
    


    def get_informative_labels(self, conversation):
        # this label is for captioning and qa task, no need to learn here
        return None


    def load_video(self, file):
        video_metadata = self.metadata[file]
        dropout_intervals = self.quality_dropout[file]
        cap = cv2.VideoCapture(video_metadata['path'])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames_total = math.floor(video_metadata['duration'] * self.frame_fps)
        frame_sec = [i / self.frame_fps for i in range(num_frames_total)]
        frames, cur_time, frame_index = [], 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:

                if dropout_intervals and cur_time > dropout_intervals[0][1]:
                    dropout_intervals.pop(0)

                if dropout_intervals and (dropout_intervals[0][0] <= cur_time <= dropout_intervals[0][1]):
                    frame = dropout_simultion(frame, w, h, dropout_type=dropout_intervals[0][2])

                frame = resize_and_pad_frame(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_index += 1
            cur_time += 1 / video_metadata['fps']
        

        # print(f"Dropped out {dropout_frame} out of {non_dropout_frame}, {dropout_frame/len(frames):.2f}% dropped out, {non_dropout_frame/len(frames): .2f}% non dropped out")
        cap.release()
        frames = np.array(frames)  # shape will be (T, H, W, C)
        frames = np.transpose(frames, (0, 3, 1, 2))  # Change to (T, C, H, W)
        return torch.tensor(frames)


    def get_relevance_labels(self, conversation):
        relevance_labels = list()
        for turn in conversation:
            if turn['role'] == 'stream' and turn['num_frames'] > 0:
                if turn['learn']:
                    relevance_labels += [turn['related']] 
                else:
                    relevance_labels += [-100] * turn['num_frames']
        
        return relevance_labels

    def __getitem__(self, index):
        anno = self.annos[index]
        res = *super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index
        return res

if __name__ == '__main__':
    from models.configuration_live import LiveConfigMixin
    from models.tokenization_live import build_live_tokenizer_and_update_config
    llava_config = LiveConfigMixin(frame_token_cls=False, frame_token_pooled=[1,1], frame_num_tokens=1)
    llava_tokenizer = build_live_tokenizer_and_update_config('lmms-lab/llava-onevision-qwen2-7b-ov', llava_config)


    dataset = HIHD(
        video_root="/data/yt8m/videos/",
        anno_file="datasets/HIHD/annotations/HIHD_metadata.csv",
        metadata_path="datasets/hisum/videos_metadata.json",
        frame_fps=1,
        tokenizer=llava_tokenizer
    )

    print('length of the dataset:', len(dataset))
    for i in range(0, min(1000, len(dataset)), 20):
        example = dataset[i]
        # print(reformat_example_for_debug(example))

