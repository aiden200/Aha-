import collections, math, json, copy, random, os, csv, sys
import cv2
from dataclasses import asdict
from tqdm import tqdm
import numpy as np
import torch
import transformers
from torchvision.io import read_video
from peft import PeftModel
logger = transformers.logging.get_logger('inference')

from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from models import build_model_and_tokenizer, fast_greedy_generate, parse_args
from .datasets import FastAndAccurateStreamingVideoQADataset


class LiveInferForBenchmark:
    def __init__(self, args) -> None:
        assert not (args.bf16 and args.fp16), "only one of --bf16 true and --fp16 true can be set"
        self.torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 #if args.fp16 else torch.float32
        self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, torch_dtype=self.torch_dtype, **asdict(args))
        self.model.eval()
        if 'llava' in args.llm_pretrained:
            self.image_processor = self.model.get_vision_tower().image_processor
        else:
            self.image_processor = None
        # self.model.to('cuda')

        # visual
        self.hidden_size = self.model.config.hidden_size
        if args.frame_fps > 0:
            self.set_fps(args.frame_fps)
        self.frame_resolution = self.model.config.frame_resolution
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens

        # generation
        self.system_prompt = args.system_prompt
        self.inplace_output_ids = torch.zeros(1, 200, device='cuda', dtype=torch.long)
        self.stream_end_prob_threshold = args.stream_end_prob_threshold
        self.response_min_interval_frames = args.response_min_interval_frames
        self.threshold_z = args.threshold_z
        self.first_n_frames_no_generate = args.first_n_frames_no_generate
        self.running_list_length = args.running_list_length
        self.stream_end_score_sum_threshold = args.stream_end_score_sum_threshold
        self.score_heads = args.score_heads.split(',')
        self.consecutive_n_frames_threshold = args.consecutive_n_frames_threshold
        print(f'score heads: {self.score_heads}')

        if int(self.threshold_z is not None) + int(self.stream_end_prob_threshold is not None) + int(self.stream_end_score_sum_threshold is not None) != 1:
            raise ValueError(f'only one of --stream_end_prob_threshold, --threshold_z and --stream_end_score_sum_threshold can be set. However, they are: {self.stream_end_prob_threshold}, {self.threshold_z}, {self.stream_end_score_sum_threshold}')
        if self.threshold_z is not None and self.first_n_frames_no_generate is None:
            raise ValueError('--first_n_frames_no_generate must be set when --threshold_z is set')

        self.remove_assistant_turns = args.remove_assistant_turns

        self.eos_token_id = self.model.config.eos_token_id
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], return_tensors='pt').to('cuda')
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to('cuda')
        self.repetition_penalty = args.repetition_penalty

        self.reset()

    def set_fps(self, fps=None, frame_interval=None):
        assert fps is not None or frame_interval is not None
        assert not (fps is not None and frame_interval is not None)
        if fps is not None:
            self.frame_fps = fps
            self.frame_interval = 1 / self.frame_fps
        else:
            self.frame_interval = frame_interval
            self.frame_fps = 1 / self.frame_interval

    # DEPRECATED
    def _call_for_response(self, video_time, query):
        raise ValueError("Depreicated, please look at original MMDuet repository https://github.com/yellow-binary-tree/MMDuet/blob/main/test/inference.py")

    # DEPRECATED
    def _call_for_streaming(self):
        raise ValueError("Depreicated, please look at original MMDuet repository https://github.com/yellow-binary-tree/MMDuet/blob/main/test/inference.py")

    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.frame_idx = 0
        self.last_role = 'system'
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        self.past_key_values = None
        self.debug_data_list = list()
        self.generated_token_ids = list()
        self.num_frames_no_reply = 0
        self.stream_end_prob_list = list()
        self.stream_end_score_sum = 0
        self.consecutive_n_frames = 0

    @torch.no_grad()
    def load_video(self, video_path):
        self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0]
        if self.image_processor is not None:
            self.video_tensor = self.image_processor.preprocess(self.video_tensor, return_tensors='pt')['pixel_values'].to('cuda').to(self.torch_dtype)
        else:
            self.video_tensor = self.video_tensor.to('cuda').to(self.torch_dtype)
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')

    def input_video_stream(self, video_frames):
        """
        input all video to video_frames at a time
        video_frames: input to visual encoder, after preprocessor
        """
        torch.cuda.empty_cache()     # prevent oov on small gpus
        if self.image_processor is not None:
            video_frames = self.image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values'].to('cuda').to(self.torch_dtype)
        else:
            video_frames = video_frames.to('cuda').to(self.torch_dtype)

        # encode the video frames in batches to prevent oov
        batch_size = 32
        for batch_i in range(0, math.ceil(len(video_frames) / batch_size)):
            video_frames_batch = video_frames[batch_i*batch_size: batch_i*batch_size+batch_size]
            frame_embeds = self.model.visual_embed(video_frames_batch).split(self.frame_num_tokens)
            self.frame_embeds_queue.extend([((r + batch_i * batch_size) / self.frame_fps, f.to('cpu')) for r, f in enumerate(frame_embeds)])
        del frame_embeds
        torch.cuda.empty_cache()     # prevent oov on small gpus?

    def input_query_stream(self, conversation):
        for turn in conversation:
            if turn['role'] == 'user':
                self.query_queue.append((turn['time'], turn['content']))

    def _encode_frame(self):
        """
        returns: informative_score, relevance_score
        """
        if not self.frame_embeds_queue:
            return None, None

        video_time, frame_embeds = self.frame_embeds_queue.popleft()
        if not self.past_key_values:
            self.last_ids = self._start_ids
        elif self.last_role == 'assistant' and not self.remove_assistant_turns:
            self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
        else:       # last_role is stream, now we just input another frame
            self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        inputs_embeds = torch.cat([
            self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
            frame_embeds.view(1, -1, self.hidden_size).to(self.last_ids.device),
        ], dim=1)
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values, return_dict=True)
        self.past_key_values = outputs.past_key_values
        self.frame_idx += 1
        self.num_frames_no_reply += 1
        informative_score = outputs.informative_logits[0,-1].softmax(dim=-1)[1].item()
        relevance_score = outputs.relevance_logits[0,-1].softmax(dim=-1)[1].item()
        self.last_role = 'stream'
        return {"informative_score": informative_score, "relevance_score": relevance_score}

    def _encode_query(self):
        query_time, query = self.query_queue.popleft()
        self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=self.last_role == 'stream', add_stream_prompt=True, return_tensors='pt').to('cuda')
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, use_cache=True, return_dict=True)
        self.past_key_values = outputs.past_key_values
        self.last_ids = outputs.logits[:, -1:].argmax(dim=-1)
        self.last_role = 'user'

    def _generate_response(self):
        self.last_ids = self._added_stream_generation_ids
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        output_ids, past_key_values, self.generated_token_ids = fast_greedy_generate(
            model=self.model, inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, eos_token_id=self.eos_token_id, inplace_output_ids=self.inplace_output_ids,
            repetition_penalty=self.repetition_penalty, generated_token_ids=self.generated_token_ids
        )

        if not self.remove_assistant_turns:
            self.past_key_values = past_key_values
            self.last_ids = output_ids[:, -1:]
        else:
            self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        self.num_frames_no_reply = 0
        self.last_role = 'assistant'
        return response

    @torch.no_grad()
    def inference(self):
        model_response_list = [{'time': q[0], 'content': q[1], 'role': 'user'} for q in self.query_queue]
        while self.frame_embeds_queue:
            # 1. check if a user query is at current time
            if self.query_queue and self.video_time >= self.query_queue[0][0]:
                self._encode_query()

            # 2. input a frame, and update the scores list
            video_scores = self._encode_frame()
            self.debug_data_list.append(dict(time=self.video_time, **video_scores))

            # 3. check the scores, if need to generate a response
            need_response = False
            stream_end_score = sum([v for k, v in video_scores.items() if k in self.score_heads])
            self.stream_end_prob_list.append(stream_end_score)
            self.stream_end_score_sum += stream_end_score
            if isinstance(self.running_list_length, int) and self.running_list_length > 0:
                self.stream_end_prob_list = self.stream_end_prob_list[-self.running_list_length:]
            if self.stream_end_score_sum_threshold is not None and self.stream_end_score_sum > self.stream_end_score_sum_threshold:
                need_response = True
                self.stream_end_score_sum = 0
            if self.stream_end_prob_threshold is not None and stream_end_score > self.stream_end_prob_threshold:
                need_response = True

            # 4. record the responses
            if need_response:
                response = self._generate_response()
                model_response_list.append({'time': self.video_time, 'content': response, 'role': 'assistant'})
                self.num_frames_no_reply = 0
                self.consecutive_n_frames = 0
            else:
                response = None

            # 5. update the video time
            self.video_time += 1 / self.frame_fps

        return sorted(model_response_list, key=lambda x: x['time'])


class DoNothingDataCollator:
    def __call__(self, batch):
        # Since batch size is 1, just return the first (and only) element
        return batch[0]


def round_numbers(data, n):
    if isinstance(data, list):
        return [round_numbers(d, n) for d in data]
    elif isinstance(data, dict):
        return {k: round_numbers(v, n) for k, v in data.items()}
    elif isinstance(data, float):
        return round(data, n)
    return data







def load_video_for_testing(video_file, return_true_frames=False):
    output_fps=2
    output_resolution=384
    # max_num_frames=100
    max_num_frames = None
    pad_color = (0, 0, 0)
    cap = cv2.VideoCapture(video_file)
    # Get original video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / input_fps
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = output_height = output_resolution

    output_fps = output_fps if output_fps > 0 else max_num_frames / video_duration
    num_frames_total = math.ceil(video_duration * output_fps)
    frame_sec = [i / output_fps for i in range(num_frames_total)]
    frame_list, cur_time, frame_index = [], 0, 0
    true_frame_index = 0
    true_frame_index_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
            if input_width > input_height:
                # Landscape video: scale width to the resolution, adjust height
                new_width = output_resolution
                new_height = int((input_height / input_width) * output_resolution)
            else:
                # Portrait video: scale height to the resolution, adjust width
                new_height = output_resolution
                new_width = int((input_width / input_height) * output_resolution)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            # pad the frame
            canvas = cv2.copyMakeBorder(
                resized_frame,
                top=(output_height - new_height) // 2,
                bottom=(output_height - new_height + 1) // 2,
                left=(output_width - new_width) // 2,
                right=(output_width - new_width + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_color
            )
            if return_true_frames:
                true_frame_index_list.append(true_frame_index)
            
            frame_list.append(np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)))
            frame_index += 1
        if max_num_frames and len(frame_list) >= max_num_frames:
            break
        cur_time += 1 / input_fps
        true_frame_index += 1
    cap.release()

    if return_true_frames:
        return torch.tensor(np.stack(frame_list)), output_fps, video_duration, true_frame_index_list

    return torch.tensor(np.stack(frame_list)), output_fps, video_duration






if __name__ == '__main__':

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

    system_prompt="A multimodal AI assistant is helping users with some activities. \
        Below is their conversation, interleaved with the list of video frames received by the assistant."


    args = parse_args('test')
    if args.skip_eval:
        print("Skipping evaluation")
        # We don't want to reevaluate
        sys.exit() 
    print(args)
    infer = LiveInferForBenchmark(args)
    
    
    if args.test_dataset == "tvsum":
        with open(args.video_metadata_file, 'r') as f:
            data = json.load(f)
        
        captions = {}
        with open(args.caption_metadata_file, 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)  # Skip the header row
            for row in reader:
                vid_category_code, vid_id, caption, url, length = row
                captions[vid_id] = {
                    "query": caption,
                }

        f_out = open(args.output_fname, 'w')
        results = []

        for video_name_with_extension in tqdm(data):
            video_uuid = video_name_with_extension[:-4]
            video_path = os.path.join(args.input_dir, video_name_with_extension)
            query = random.choice(query_templates) % captions[video_uuid]["query"]

            video_frames, fps, video_duration, true_frames_list = load_video_for_testing(video_path, return_true_frames=True)    

            conversation = list()
            conversation.append({"role": "system", "content": system_prompt})
            conversation.append({'role': 'user', 'content': query, 'time': 0})


            infer.reset()
            print(f"num frames and fps for {video_uuid}: {len(video_frames)}, {fps}")
            infer.set_fps(fps=fps)
            infer.input_video_stream(video_frames)
            infer.input_query_stream(conversation)
            model_response_list = infer.inference()
            res = {'video_uuid': video_uuid, 'model_response_list': model_response_list, 'video_duration': video_duration, 'true_frames_list': true_frames_list}
            res['debug_data'] = round_numbers(infer.debug_data_list, 3)
            results.append(res)
        f_out.write(json.dumps(results, indent=4))
        f_out.flush()

    
