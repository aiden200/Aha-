import collections, math, json, copy, random, os, csv, sys, yaml
import cv2
from dataclasses import asdict
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import torch
import transformers
from torchvision.io import read_video
from peft import PeftModel
import ast
import h5py



logger = transformers.logging.get_logger('inference')

from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


from models import build_model_and_tokenizer, fast_greedy_generate, parse_args
from .datasets import FastAndAccurateStreamingVideoQADataset
from test.arl_scout.prepare_data import generate_plot


class LiveInferForBenchmark:
    def __init__(self, args) -> None:
        assert not (args.bf16 and args.fp16), "only one of --bf16 true and --fp16 true can be set"

        self.peft_model_id = "aiden200/aha"  
        self.torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 #if args.fp16 else torch.float32
        self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, torch_dtype=self.torch_dtype, **asdict(args))
        self.model.eval()
        self.model.currently_training = False
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

        self.uncertainty_wait_threshold = args.uncertainty_wait_threshold
        self.max_wait_frames = args.max_wait_frames
        self.uncertainty_lock = 0

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
        returns: informative_score, relevance_score, uncertainty_score
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
        # print(outputs)
        self.past_key_values = outputs.past_key_values
        self.frame_idx += 1
        self.num_frames_no_reply += 1
        informative_score = outputs.informative_logits[0,-1].softmax(dim=-1)[1].item()
        if outputs.relevance_logits.shape[-1] == 2:
            relevance_score = outputs.relevance_logits[0, -1].softmax(dim=-1)[1].item()
        else:
            relevance_score = outputs.relevance_logits[0, -1].item()
        uncertainty_score = torch.exp(outputs.uncertainty[0, -1]).item()
        self.last_role = 'stream'
        return {"informative_score": informative_score, "relevance_score": relevance_score}, uncertainty_score

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
            video_scores, uncertainty_score = self._encode_frame()
            self.debug_data_list.append(dict(time=self.video_time, **video_scores,uncertainty_score=uncertainty_score))
            # self.debug_data_list.append(dict(time=self.video_time, **video_scores))

            # 3. check the scores, if need to generate a response
            # We may want the importance head to be the dictator, since we are training relevance with MSE loss
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
            
            
            # if self.uncertainty_lock:
            #     # we add on how many frames we locked down
            #     self.uncertainty_lock += 1
            
            # # If uncertainty is high, wait
            # if need_response and self.uncertainty_wait_threshold is not None and \
            #     uncertainty_score > self.uncertainty_wait_threshold:

            #     # Another way we can define the threshold
            #     # if self.num_frames_no_reply > self.max_wait_frames:
            #     #     need_response = True
            #     # else:
            #     #     need_response = False

            #     if self.uncertainty_lock < self.max_wait_frames:
            #         need_response = False
            #     else:
            #         need_response = True
            #         self.uncertainty_lock = 0

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



def load_individual_frames_for_testing(frame_folder, output_fps=1):
    output_resolution=384
    frame_names = os.listdir(frame_folder)
    pad_color = (0, 0, 0)
    frame_count = len(frame_names)
    video_duration = frame_count
    
    frame_list = []
    output_width = output_height = output_resolution
    
    for i in tqdm(range(len(frame_names))):
        full_path = os.path.join(frame_folder, f"frame{i:03d}.jpg")
        frame = Image.open(full_path)
        width, height = frame.size
        if width > height:
            new_width = output_resolution
            new_height = int((height / width) * output_resolution)
        else:
            new_height = output_resolution
            new_width = int((width / height) * output_resolution)
        
        resized_frame = frame.resize((new_width, new_height))
        left = (output_width - new_width) // 2
        right = (output_width - new_width + 1) // 2
        top = (output_height - new_height) //2 
        bottom = (output_height - new_height + 1) //2
        canvas = ImageOps.expand(resized_frame, border=(left, top, right, bottom), fill=pad_color)
        canvas_np = np.array(canvas)
        canvas_np = np.transpose(canvas_np, (2, 0, 1))
        frame_list.append(canvas_np)
    
    
    return torch.tensor(np.stack(frame_list)), output_fps, video_duration
        


def load_video_for_testing(video_file, output_fps=2, return_true_frames=False, max_num_frames=None):
    output_resolution=384
    # max_num_frames = None
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

    if not frame_list:
        return None, None, None, None

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

    # So we can get through the DDP Print statements
    os.environ['RANK'] = "0"

    args = parse_args('test')
    if args.skip_eval:
        print("Skipping evaluation")
        # We don't want to reevaluate
        sys.exit() 
    
    all_arg_fields = ["is_online_model", "grounding_mode", "stream_end_prob_threshold", 
                      "response_min_interval_frames", "start_idx", "end_idx", "stream_end_score_sum_threshold",
                      "remove_assistant_turns", "score_heads", "skip_eval", "input_dir", "test_fname", "output_fname", 
                      "test_dataset", "caption_metadata_file", "video_metadata_file", "hisum_h5_file", "anno_file",
                      "frame_fps", "max_num_frames", "func"]


    print(args)

    # infer = LiveInferForBenchmark(args)
    
    
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

        results = []
        infer = LiveInferForBenchmark(args)


        for video_name_with_extension in tqdm(data):
            video_uuid = video_name_with_extension[:-4]
            video_path = os.path.join(args.input_dir, video_name_with_extension)
            query = random.choice(query_templates) % captions[video_uuid]["query"]

            # max_num_frames=100
            max_num_frames = None
            video_frames, fps, video_duration, true_frames_list = load_video_for_testing(video_path, output_fps=args.frame_fps, return_true_frames=True, max_num_frames=max_num_frames)    
            if video_frames == None:
                continue
            conversation = list()
            conversation.append({"role": "system", "content": system_prompt})
            conversation.append({'role': 'user', 'content': query, 'time': 0})


            infer.reset()
            # print(f"num frames and fps for {video_uuid}: {len(video_frames)}, {fps}")
            infer.set_fps(fps=fps)
            infer.input_video_stream(video_frames)
            infer.input_query_stream(conversation)
            model_response_list = infer.inference()
            res = {'video_uuid': video_uuid, 'model_response_list': model_response_list, 'video_duration': video_duration, 'true_frames_list': true_frames_list}
            
            res['debug_data'] = round_numbers(infer.debug_data_list, 3)
            # print(res['debug_data'])
            results.append(res)
        f_out = open(args.output_fname, 'w')
        f_out.write(json.dumps(results, indent=4))
        f_out.flush()


    elif args.test_dataset == "hisum":
        
        with open(args.video_metadata_file, 'r') as f:
            data = json.load(f)
        
        results = []
        anno_path = os.path.join(args.anno_file) # .train split file
        h5_file = os.path.join(args.hisum_h5_file) #.h5 file
        hisum_metadata = os.path.join(args.caption_metadata_file) #.json file with metadata
        assert os.path.exists(anno_path) and os.path.exists(h5_file) and os.path.exists(hisum_metadata)
        with open(anno_path, "r") as f:
            # videos = json.load(f)["test_keys"]
            videos = json.load(f)["test_keys"]
        
        video_info = {}
        infer = LiveInferForBenchmark(args)
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
        
        with h5py.File(h5_file, "r") as hdf:
            success_vids = 0
            all_files = os.listdir(args.input_dir)
            for video_id in tqdm(videos):


                if video_id in video_info:
                    video_filepath = f"{video_info[video_id]['youtube_id']}.mp4"
                    try:
                        # checking if we managed to download the video
                        if video_filepath in all_files:
                            success_vids += 1
                            importance_scores = list(hdf[video_id]["gtscore"])
                            categories = video_info[video_id]["categories"]
                            caption = video_info[video_id]["caption"]
                            
                            
                            video_uuid = video_filepath[:-4]
                            video_path = os.path.join(args.input_dir, video_filepath)
                            query = random.choice(query_templates) % caption

                            # max_num_frames=100
                            max_num_frames = None
                            video_frames, fps, video_duration, true_frames_list = load_video_for_testing(video_path, output_fps=args.frame_fps, return_true_frames=True, max_num_frames=max_num_frames)    
                            if video_frames == None:
                                continue
                            conversation = list()
                            conversation.append({"role": "system", "content": system_prompt})
                            conversation.append({'role': 'user', 'content': query, 'time': 0})


                            infer.reset()
                            # print(f"num frames and fps for {video_uuid}: {len(video_frames)}, {fps}")
                            infer.set_fps(fps=fps)
                            infer.input_video_stream(video_frames)
                            infer.input_query_stream(conversation)
                            model_response_list = infer.inference()
                            res = {"categories": video_info[video_id]["categories"],'h5_identifier': video_id, 'video_uuid': video_uuid, 'model_response_list': model_response_list, 'video_duration': video_duration, 'true_frames_list': true_frames_list}
                            res['debug_data'] = round_numbers(infer.debug_data_list, 3)
                            results.append(res)
                    except Exception as e:
                        print(f"Exception on video: {video_filepath} with exception: {e}")
        
        
        
        f_out = open(args.output_fname, 'w')
        f_out.write(json.dumps(results, indent=4))
        f_out.flush()
    
    elif args.test_dataset == "arl_scout":
        frame_folder = args.input_dir
        output_file = args.output_fname
        parent_dir = os.path.dirname(output_file)
        skip = True

        if not skip:
            caption = "what objects are in this room?"
            query = random.choice(query_templates) % caption
            # max_num_frames=100
            max_num_frames = None
            video_frames, fps, video_duration = load_individual_frames_for_testing(frame_folder)
            
            infer = LiveInferForBenchmark(args)
            if video_frames == None:
                raise ValueError("Error, no video frames")
            conversation = list()
            conversation.append({"role": "system", "content": system_prompt})
            conversation.append({'role': 'user', 'content': query, 'time': 0})


            infer.reset()
            infer.set_fps(fps=fps)
            infer.input_video_stream(video_frames)
            infer.input_query_stream(conversation)
            model_response_list = infer.inference()
            model_response_formated = {}
            for response in model_response_list:
                if response["role"] == "assistant":
                    model_response_formated[response["time"]] = response["content"]  
            results = round_numbers(infer.debug_data_list, 3)
            with open(output_file, "w") as f:
                json.dump(results, f)
        else:
            with open(output_file, "r") as f:
                results = json.load(f)
        
        times = [d["time"] for d in results]
        informative_scores = [d["informative_score"] for d in results]
        relevance_scores = [d["relevance_score"] for d in results]
        uncertainty_scores = [d["uncertainty_score"] for d in results]
        # relevance_scores = np.convolve(relevance_scores, np.ones(5)/5, mode='same')

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(times, informative_scores, label="Informative Score")
        plt.plot(times, relevance_scores, label="Relevance Score")
        plt.plot(times, uncertainty_scores, label="Uncertainty Score")

        plt.xlabel("Time")
        plt.ylabel("Score")
        plt.title("Scores over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(parent_dir, "visualization.png"))

        os.makedirs(os.path.join(parent_dir, "stiched"), exist_ok=True)

        stiched_img_paths = []
        if not skip:
            print(model_response_formated)
        
        for idx in range(len(results)):
            # Load frame
            frame_path = os.path.join(frame_folder, f"frame{idx:03d}.jpg")
            frame_img = Image.open(frame_path).convert("RGB")
            
            agent_response = None
            if not skip and idx in model_response_formated:
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
    
    else:
        with open("paths.yaml", "r") as f:
            dataset_args = yaml.safe_load(f)[args.test_dataset]
        
        if "threshold" in dataset_args:
            print(f"Dataset: {args.test_dataset} is using grid search param: threshold {dataset_args['threshold']}")
            args.stream_end_score_sum_threshold = dataset_args["threshold"]

        dataset = FastAndAccurateStreamingVideoQADataset(
            data_file=args.test_fname, video_base_folder=args.input_dir,
            start_idx=args.start_idx, end_idx=args.end_idx,
            output_fps=args.frame_fps, output_resolution=args.frame_resolution, max_num_frames=args.max_num_frames,
            time_instruction_format=args.time_instruction_format, system_prompt=args.system_prompt
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=DoNothingDataCollator())
        f_out = open(args.output_fname, 'w')

        
        infer = LiveInferForBenchmark(args)
        if args.is_online_model:
            if not args.grounding_mode:
                # Youcook2
                

                for data_i, data in enumerate(tqdm(dataloader)):
                    question_id, video_frames, conversation, fps, video_duration = data
                    if question_id is None: continue
                    infer.reset()
                    # print(f"num frames and fps for {question_id}: {len(video_frames)}, {fps}")
                    infer.set_fps(fps=fps)
                    infer.input_video_stream(video_frames)
                    infer.input_query_stream(conversation)
                    model_response_list = infer.inference()
                    res = {'question_id': question_id, 'model_response_list': model_response_list, 'video_duration': video_duration}
                    res['debug_data'] = round_numbers(infer.debug_data_list, 3)
                    f_out.write(json.dumps(res) + '\n')
                    if data_i % 5 == 0:
                        f_out.flush()
                f_out.close()

            else:
                infer.first_n_frames_no_generate = 100000       # so the generation process is never called, we just want `relevance_score` results
                for data_i, data in enumerate(tqdm(dataloader)):
                    question_id, video_frames, conversation, fps, video_duration = data
                    if question_id is None: continue
                    infer.reset()
                    # print(f"num frames and fps for {question_id}: {len(video_frames)}, {fps}")
                    infer.set_fps(fps=fps)
                    infer.input_video_stream(video_frames)
                    infer.input_query_stream(conversation)
                    model_response_list = infer.inference()
                    res = {'question_id': question_id, 'model_response_list': model_response_list, 'video_duration': video_duration}
                    res['debug_data'] = round_numbers(infer.debug_data_list, 3)
                    f_out.write(json.dumps(res) + '\n')
                    if data_i % 5 == 0:
                        f_out.flush()
                f_out.close()

        else:
            # llava onevision baseline
            tokenizer, model, image_processor, max_length = load_pretrained_model(args.llm_pretrained, None, "llava_qwen", device_map="auto", attn_implementation=args.attn_implementation)  # Add any other thing you want to pass in llava_model_args
            model.eval()

            if args.lora_pretrained is not None:
                print(f"loading lora ckpt from {args.lora_pretrained}, and setting mm_spatial_pool_stride to {args.video_pooling_stride}")
                model = PeftModel.from_pretrained(model, args.lora_pretrained, is_trainable=False)
                model.config.mm_spatial_pool_stride = args.video_pooling_stride

            f_out = open(args.output_fname, 'w')
            for data_i, data in enumerate(tqdm(dataloader)):
                question_id, video_frames, conversation, fps, video_duration = data
                if question_id is None: continue
                conv_template = "qwen_1_5"
                original_question = [e['content'] for e in conversation if e['role'] == 'user'][0]
                question = f"{DEFAULT_IMAGE_TOKEN}\n{original_question}"
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                if data_i < 5:
                    print(f'model input at example {data_i}: {prompt_question}')
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                image_sizes = [frame.size() for frame in video_frames]
                image_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(model.device)
                modalities = ["video"] * len(video_frames)
                cont = model.generate(
                    input_ids,
                    images=[image_tensor],
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=512,
                    modalities=modalities,
                )
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
                res = {'question_id': question_id, 'model_response': text_outputs, 'question': original_question, 'video_duration': video_duration}
                f_out.write(json.dumps(res) + '\n')
                if data_i % 10 == 0:
                    f_out.flush()
            f_out.close()
    
