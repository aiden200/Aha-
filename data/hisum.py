import json, os, shutil, csv
import cv2
import h5py
from tqdm import tqdm, trange
import math
import random
import numpy as np
from typing import DefaultDict

from transformers.utils import logging
from .stream import StreamMixIn
from .utils import reformat_example_for_debug, DictWithTo
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

        #TODO caption data path, conversation generation

        # Annos is the annotation .json file that we have with.
        # .h5 file that generates annotation
        # Self.metadata is something that the model generates
        # We also have the video0001 -> yt_name conversion
        for anno in tqdm(annos):
            curr_info = annos[anno]
            video_uid = curr_info['video_uid']
            caption_data =self.captions[video_uid]

            # to match metadata parsing
            video_uid = video_uid + ".mp4"
            
            if video_uid not in self.metadata:
                logger.warning(f'Missmatch in captions and annotations: {video_uid}')
                continue
            duration = self.metadata[video_uid]['duration']
            conversation, current_frame = list(), 0
            conversation.append({'role': 'user', 'content': random.choice(self.query_templates) % caption_data['query'], 'learn': False})

            for i in range(len(curr_info["importance_scores"])):
                conversation.append({'role': 'stream', 'num_frames': 1, 'learn': True, 'related': curr_info["importance_scores"][i]})
            last_frame = math.floor(duration * self.frame_fps)
            
            self.annos.append({
                'conversation': conversation,
                'load_ranges': {video_uid: range(0, last_frame)}
            })
        print(f'Dataset {self.__class__.__name__} has {len(self)} examples. Example data: {reformat_example_for_debug(self[0])}')


    def get_annos(self) -> dict:
        annotations = {}
        anno_path = os.path.join(self.anno_file) # .train split file
        h5_file = os.path.join(self.hisum_h5_file) #.h5 file
        hisum_metadata = os.path.join(self.hisum_metadata) #.json file with metadata
        assert os.path.exists(anno_path) and os.path.exists(h5_file) and os.path.exists(hisum_metadata)
        with open(anno_path, "r") as f:
            videos = json.load(f)["train_keys"]
        
        video_info = {}

        with open(hisum_metadata, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                video_info[row["yt8m_file"]] = {
                    "caption" : row["title"],
                    "categories" : row["labels"] # are the labels commas going to affect something? try loading also need to parse
                    #TODO
                }
        
        with h5py.File(anno_path, "r") as hdf:
            for video in videos:
                importance_scores = list(hdf[video]["gtscore"])
                category = hdf[video][""]
                #TODO add category from two json files
                #TODO might as well add captions too
                annotations[video] = {
                    "importance_scores": importance_scores,
                    "categories": categories,
                    "caption": caption
                }
                

               
        return annotations
    


    def get_informative_labels(self, conversation):
        # this label is for captioning and qa task, no need to learn here
        return None


    def get_relevance_labels(self, conversation):
        relevance_labels = list()
        for turn in conversation:
            if turn['role'] == 'stream' and turn['num_frames'] > 0:
                if turn['learn']:
                    relevance_labels += [turn['related']] # this is modified, we need to store all the values now
                else:
                    relevance_labels += [-100] * turn['num_frames']
        
        return relevance_labels

    def __getitem__(self, index):
        # try:
        anno = self.annos[index]
        res = *super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index
        # except Exception as e:
        #     logger.warning(f'Error in dataset {self.anno_file} when getting index {index}: {e}')
        #     logger.warning(f'Using a random data instead.')
        #     res = self.__getitem__(random.choice(list(range(len(self)))))
        return res

if __name__ == '__main__':
    from models.configuration_live import LiveConfigMixin
    from models.tokenization_live import build_live_tokenizer_and_update_config
    llava_config = LiveConfigMixin(frame_token_cls=False, frame_token_pooled=[1,1], frame_num_tokens=1)
    llava_tokenizer = build_live_tokenizer_and_update_config('lmms-lab/llava-onevision-qwen2-7b-ov', llava_config)


    dataset = HiSumDataset(
        video_root="/data/yt8m/videos",
        anno_file="datasets/hisum/annotations/train.json",
        metadata_path="datasets/hisum/videos_metadata.json",
        hisum_h5_file="datasets/hisum/annotations/mr_hisum.h5",
        hisum_metadata="datasets/hisum/annotations/mr_hisum_metadata.csv",
        frame_fps=0.5,
        tokenizer=llava_tokenizer
    )

    print('length of the dataset:', len(dataset))
    for i in range(0, min(1000, len(dataset)), 20):
        example = dataset[i]
        print(reformat_example_for_debug(example))
