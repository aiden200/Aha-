from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class LiveTrainingArguments(TrainingArguments):
    grounding_mode: bool = False        # if set, only output probs, never generate reply
    live_version: str = 'live1+'
    input_dir: str = 'dataset/tvsum/ydata-tvsum50-v1_1/video'
    dataset_config: str = None
    stream_loss_weight: float = 1.0
    llm_pretrained: str = 'lmms-lab/llava-onevision-qwen2-7b-ov'
    vision_pretrained: str = 'google/siglip-large-patch16-384'
    lora_pretrained: str = None
    lora_modules: str = "model\.layers.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    lora_r: int = 16
    lora_alpha: int = 32
    finetune_modules: list[str] = field(default_factory=lambda: ['connector', 'mm_projector', 'response_head', 'lm_head', 'informative_head', 'relevance_head', 'uncertainty_head'])
    frame_fps: float = 2
    frame_token_cls: bool = False
    frame_token_pooled: list[int] = field(default_factory=lambda: [7,7])
    frame_num_tokens: int = 49
    video_pooling_stride: int = 4
    frame_resolution: int = 384
    use_cache: bool = False
    embed_mark: str = '2fps_384_1+3x3'
    v_placeholder: str = '<image>'
    max_num_frames: int = 100
    augmentation: bool = False
    attn_implementation: str = 'flash_attention_2'
    output_dir: str = 'outputs/debug'

    first_n_frames_no_generate: int = 0 # We want to be mindful of first few arguments
    quantization: bool = False
    push_to_hub: bool = True
    max_grad_norm: float = 1.0



@dataclass
class LiveTestArguments(LiveTrainingArguments):
    system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
    live_version: str = 'test'
    is_online_model: bool = True
    grounding_mode: bool = False        # if set, only output probs, never generate reply
    repetition_penalty: float = None
    stream_end_prob_threshold: float = None
    response_min_interval_frames: int = None
    threshold_z: float = None
    first_n_frames_no_generate: int = 0
    consecutive_n_frames_threshold: int = 1
    running_list_length: int = 20
    start_idx: int = 0
    end_idx: int = None
    time_instruction_format: str = None
    stream_end_score_sum_threshold: float = None
    remove_assistant_turns: bool = False        # if True, do not add assistant-generated content to input context (kv_cache)
    score_heads: str = 'relevance_score,informative_score'       # a list of score names, seperated with comma. e.g.: `relevance_score,informative_score`
    skip_eval: bool = False # skips evaluation
    uncertainty_wait_threshold: float = 0.0 # based on log variance, or 1.0 if using variance
    max_wait_frames: int = 3 # maximum frames to wait before forcing a response, no matter how high uncertainty is
    
    # Evaluation specific arguments.
    input_dir: str = ''
    test_fname: str = ''
    output_fname: str = ''
    test_dataset: str = "" # the type of dataset 
    caption_metadata_file: str = "" # the caption file if applicable
    video_metadata_file: str = '' # the video metadata file if applicable
    hisum_h5_file: str = "" # hisum file
    anno_file: str = ""
    no_query: bool = False # For ablations - don't input query if set to True


def get_args_class(args_version: str):
    if args_version == 'train':
        return LiveTrainingArguments
    elif args_version == 'test':
        return LiveTestArguments
    raise NotImplementedError
