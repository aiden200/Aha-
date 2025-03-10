import json, os, shutil, csv
import cv2
from tqdm import tqdm, trange
import math
import random
import numpy as np
from typing import DefaultDict

from transformers.utils import logging
from .stream import StreamMixIn
from .utils import reformat_example_for_debug, DictWithTo
logger = logging.get_logger(__name__)


class HumanIntuitionDataset(StreamMixIn):
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

        # Annos is the annotation .json file that we need to create, not the one the model automatically generates.
        # Self.metadata is something that the model generates
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
        # load tsv file and get average importance scores
        anno_path = os.path.join(self.anno_file)
        assert os.path.exists(anno_path)
        vid_count = DefaultDict(int)
        annotations = {}
        with open(anno_path, 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                video_id = row[0]
                category_code = row[1]
                importance_scores = np.array(list(map(int, row[2].split(','))), dtype=np.float64)
                if video_id not in annotations:
                    annotations[video_id] = {
                        "importance_scores": importance_scores,
                        "video_uid": video_id
                    }
                else:
                    annotations[video_id]["importance_scores"]  += importance_scores
                
                vid_count[video_id] += 1
        
        for video in annotations:
            annotations[video]["importance_scores"] /= vid_count[video]
            # Normalizing the score, the maximum score is 5
            annotations[video]["importance_scores"]  /= 5.0
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

    dataset = HumanIntuitionDataset(
        video_root='datasets/tvsum/ydata-tvsum50-v1_1/video',
        anno_file='datasets/tvsum/annotations/train.json', 
        metadata_path='datasets/tvsum/videos_metadata.json', # This is automatically generated
        system_prompt='This is a system prompt.', tokenizer=llava_tokenizer,
        frame_fps=0.5, max_num_frames=120
    )

    print('length of the dataset:', len(dataset))
    for i in range(0, min(1000, len(dataset)), 20):
        example = dataset[i]
        print(reformat_example_for_debug(example))
