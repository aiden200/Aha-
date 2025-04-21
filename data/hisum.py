import json, os, shutil, csv
import cv2
import h5py
from tqdm import tqdm, trange
import math
import random
import numpy as np
import time
from typing import DefaultDict
import ast

from transformers.utils import logging
from .stream import StreamMixIn
from .utils import reformat_example_for_debug, DictWithTo, is_valid_video_cap

logger = logging.get_logger(__name__)


class HiSumDataset(StreamMixIn):
    query_templates = [
        "%s",
        "%s",
        "What segment of the video addresses the topic '%s'?",
        "At what timestamp can I find information about '%s' in the video?",
        "Can you highlight the section of the video that pertains to '%s'?",
        "Which moments in the video discuss '%s' in detail?",
        "Identify the parts that mention '%s'.",
        "Where in the video is '%s' demonstrated or explained?",
        "What parts are relevant to the concept of '%s'?",
        "Which clips in the video relate to the query '%s'?",
        "Can you point out the video segments that cover '%s'?",
        "What are the key timestamps in the video for the topic '%s'?"
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        annos, self.annos = self.annos, list()


        # Annos is the annotation .json file that we have with.
        # .h5 file that generates annotation
        # Self.metadata is something that the model generates
        # We also have the video0001 -> yt_name conversion
        for anno in tqdm(annos):
            curr_info = annos[anno]
            video_uid = curr_info['video_uid']
            caption = curr_info["caption"]
            importance_scores = curr_info["importance_scores"]

            # to match metadata parsing
            video_uid = video_uid + ".mp4"
            
            if video_uid not in self.metadata:
                logger.warning(f'Missmatch in captions and annotations: {video_uid}')
                continue
            if not self.metadata[video_uid]["valid"]:
                logger.warning(f"Video {video_uid} not loaded, possibly corrupted")
                continue
            duration = self.metadata[video_uid]['duration']
            conversation, current_frame = list(), 0
            conversation.append({'role': 'user', 'content': random.choice(self.query_templates) % caption, 'learn': False})
            
            # One score per duration
            for i in range(len(importance_scores)):
                conversation.append({'role': 'stream', 'num_frames': 1, 'learn': True, 'related': importance_scores[i]})
            

            final_frame = math.floor(duration * self.frame_fps)
            if final_frame < len(conversation):
                conversation = conversation[:final_frame + 1]
            
            self.annos.append({
                'conversation': conversation,
                'load_ranges': {video_uid: range(0, final_frame)}
            })
        print(f'Dataset {self.__class__.__name__} has {len(self)} examples. Example data: {reformat_example_for_debug(self[0])}')
    

    def get_annos(self) -> dict:
        annotations = {}
        anno_path = os.path.join(self.anno_file) # .train split file
        h5_file = os.path.join(self.hisum_h5_file) #.h5 file
        hisum_metadata = os.path.join(self.hisum_metadata) #.json file with metadata
        # print(anno_path, h5_file, hisum_metadata)
        assert os.path.exists(anno_path) and os.path.exists(h5_file) and os.path.exists(hisum_metadata)
        with open(anno_path, "r") as f:
            random.seed(22)
            videos = json.load(f)["train_keys"]
            random.shuffle(videos)
            videos = videos[:12000]
        
        video_info = {}

        with open(hisum_metadata, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    categories = ast.literal_eval(row["labels"])  # Convert string to list
                except (SyntaxError, ValueError):
                    logger.warning("Hisum failed to parse categories")
                    categories = []  # Default to empty list if parsing fails
                video_info[row["video_id"]] = {
                    "caption" : row["title"],
                    "categories" : [c for c in categories if c],
                    'youtube_id' : row["youtube_id"],
                    'yt8m_id': row["yt8m_file"],
                    'video_id': row["video_id"]
                }
        
        # s = time.time()
        
        with h5py.File(h5_file, "r") as hdf:
            success_vids = 0
            all_files = os.listdir(self.video_root)
            for video_id in tqdm(videos):
                # If we were able to obtain the caption and download the video
                # video_id is represented by video_[VID NUMBER] 

                if video_id in video_info:
                    video_filepath = f"{video_info[video_id]['youtube_id']}.mp4"
                    # checking if we managed to download the video
                    if video_filepath in all_files:
                        success_vids += 1
                        importance_scores = list(hdf[video_id]["gtscore"])
                        categories = video_info[video_id]["categories"]
                        caption = video_info[video_id]["caption"]
                        video_uid = video_info[video_id]["youtube_id"]
                        annotations[video_id] = {
                            "importance_scores": importance_scores,
                            "categories": categories,
                            "caption": caption,
                            "video_uid": video_uid, #""
                            "video_id": video_id
                        }
        print(f"Mr. HiSum loaded {success_vids} out of {len(videos)} videos")
        logger.info(f"Mr. HiSum dataset loaded {len(annotations)} out of {len(videos)} videos")

        return annotations
    


    def get_informative_labels(self, conversation):
        # this label is for captioning and qa task, no need to learn here
        return None


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


    dataset = HiSumDataset(
        video_root="/mnt/training-data/yt8m/",
        anno_file="datasets/hisum/annotations/split.json",
        metadata_path="datasets/hisum/videos_metadata.json",
        hisum_h5_file="datasets/hisum/annotations/mr_hisum.h5",
        hisum_metadata="datasets/hisum/annotations/mr_hisum_metadata.csv",
        frame_fps=1,
        tokenizer=llava_tokenizer
    )

    print('length of the dataset:', len(dataset))
    for i in range(0, min(1000, len(dataset)), 20):
        example = dataset[i]
        # print(reformat_example_for_debug(example))
