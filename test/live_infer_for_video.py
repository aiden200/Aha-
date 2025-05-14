import os, math
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from test.inference import LiveInferForBenchmark
from scipy.signal import find_peaks, savgol_filter


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

def load_video(video_file, output_fps):
    pad_color = (0, 0, 0)
    output_resolution = 384
    max_num_frames = 400

    cap = cv2.VideoCapture(video_file)
    # Get original video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / input_fps
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = output_height = output_resolution

    output_fps = output_fps if output_fps > 0 else max_num_frames / video_duration
    num_frames_total = math.floor(video_duration * output_fps)
    frame_sec = [i / output_fps for i in range(num_frames_total)]
    frame_list, original_frame_list, cur_time, frame_index = [], [], 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
            original_frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
            frame_list.append(np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)))
            frame_index += 1
        if len(frame_list) >= max_num_frames:
            break
        cur_time += 1 / input_fps
    cap.release()
    return torch.tensor(np.stack(frame_list)), original_frame_list, output_fps


class LiveInferForDemo(LiveInferForBenchmark):
    def __init__(self, args, peft_model_id, query=None):
        super().__init__(args, peft_model_id)
        # self.encode_given_query(query)
        self.system_prompt = "A multimodal AI assistant is helping users with some activities. \
        Below is their conversation, interleaved with the list of video frames received by the assistant."


    def encode_given_query(self, query):
        self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=self.last_role == 'stream', add_stream_prompt=True, return_tensors='pt').to('cuda')
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, use_cache=True, return_dict=True)
        self.past_key_values = outputs.past_key_values
        self.last_ids = outputs.logits[:, -1:].argmax(dim=-1)
        self.last_role = 'user'



    def load_one_frame(self, frame_path=None, frame_object=None):
        assert frame_path or frame_object
        pad_color = (0, 0, 0)
        output_resolution = 384
        if not frame_object:
            frame = Image.open(frame_path)
        else:
            frame = frame_object
        
        width, height = frame.size
        if width > height:
            new_width = output_resolution
            new_height = int((height / width) * output_resolution)
        else:
            new_height = output_resolution
            new_width = int((width / height) * output_resolution)
        
        resized_frame = frame.resize((new_width, new_height))
        left = (output_resolution - new_width) // 2
        right = (output_resolution - new_width + 1) // 2
        top = (output_resolution - new_height) //2 
        bottom = (output_resolution - new_height + 1) //2
        canvas = ImageOps.expand(resized_frame, border=(left, top, right, bottom), fill=pad_color)
        canvas_np = np.array(canvas)
        canvas_np = np.transpose(canvas_np, (2, 0, 1))

        video_frames = self.image_processor.preprocess([canvas_np], return_tensors='pt')['pixel_values'].to('cuda').to(self.torch_dtype)
        frame_embeds = self.model.visual_embed(video_frames).split(self.frame_num_tokens)

        self.frame_embeds_queue.append((self.video_time, frame_embeds[0].to('cpu')))
        torch.cuda.empty_cache()
        





    def input_one_frame(self):
        """
        in the interactive demo, we need to input 1 frame each time this function is called.
        to ensure that user can stop the video and input user messages.
        """
        # 1. the check query step is skipped, as all user input is from the demo page

        # 2. input a frame, and update the scores list
        video_scores, uncertainty_scores = self._encode_frame()
        ret = dict(
            frame_idx=self.frame_idx,
            time=round(self.video_time, 1),
            uncertainty_score=uncertainty_scores,
            **video_scores
        )

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
            self.num_frames_no_reply = 0
            self.consecutive_n_frames = 0
        else:
            response = None
        ret['response'] = response

        # 5. update the video time
        self.video_time += 1 / self.frame_fps

        return ret


    def input_video(self, video_path, query, fps=None):
        video_frames, original_frame_list, output_fps = load_video(video_path, output_fps=fps)    
        conversation = list()
        conversation.append({"role": "system", "content": self.system_prompt})
        conversation.append({'role': 'user', 'content': query, 'time': 0})


        # self.reset()
        self.set_fps(fps=output_fps)
        self.input_video_stream(video_frames)
        self.input_query_stream(conversation)
        model_response_list = self.inference(verbose=True, total=len(video_frames))
        results = round_numbers(self.debug_data_list, 3)
        return results, model_response_list


    def find_ticks(
            self, 
            scores, 
            fps, 
            min_separation=10,
            prominence=0.02,
            thresh=False, 
            verbose=False):
        '''
        Return a list of highlight portions
        '''
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        smoothed = savgol_filter(scores, window_length=15, polyorder=3)
        if not thresh:
            thresh = smoothed.mean() + 0.5*smoothed.std()
                       
        min_separation = 10  
        distance = int(min_separation * fps)

        peaks, props = find_peaks(
            smoothed,
            height=thresh, 
            prominence=prominence,      # only “sharp” spikes (tune as needed)
            distance=distance
        )

        # 5) convert to times (seconds)
        peak_times = peaks / fps

        if verbose:
            print("Detected spikes at:", peak_times)
        return list(peak_times)
    

    # def generate_highlight_reel(scores, input_file, output_file, window=1):
