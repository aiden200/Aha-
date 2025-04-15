#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
import copy
import random
import wandb
import os
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from dataclasses import dataclass

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from ..modeling_live import build_live, LiveMixin
from ..configuration_live import VideoHeadLiveConfigMixin


from transformers.utils import logging
logger = logging.get_logger(__name__)


class VideoHeadLiveLlavaQwenConfig(Qwen2Config, VideoHeadLiveConfigMixin):
    def __init__(self, video_pooling_stride=4, video_head_stop_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.video_pooling_stride = video_pooling_stride
        self.video_head_stop_grad = video_head_stop_grad


@dataclass
class VideoHeadCausalLMOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    lm_loss: Optional[torch.FloatTensor] = None
    video_loss: Optional[torch.FloatTensor] = None
    informative_logits: Optional[torch.FloatTensor] = None
    relevance_logits: Optional[torch.FloatTensor] = None
    uncertainty: Optional[torch.FloatTensor] = None

class VideoHeadLlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = VideoHeadLiveLlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(VideoHeadLlavaQwenModel, self).__init__(config)


class VideoHeadLiveLlavaQwenForCausalLM(Qwen2ForCausalLM, LiveMixin):
    config_class = VideoHeadLiveLlavaQwenConfig

    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
            
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        self.global_step = 0
        self.model = VideoHeadLlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.currently_training = True

        self.informative_head = nn.Linear(config.hidden_size, 2, bias=False)
        self.relevance_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.uncertainty_head = nn.Linear(config.hidden_size, 1, bias=False)

        self.post_init()
        self.vision_encoder = self.get_vision_tower()


        self.lm_loss_weight = .2
        self.video_loss_weight = 1
        self.info_loss_weight = 0.5
        self.ref_loss_weight = 8.0
        self.uncertainty_loss_weight = 0.1
        self.tv_loss_weight = 0.01
        if torch.distributed.is_initialized() and int(os.environ["RANK"]) == 0:
            print(f"using lm_loss_weight: {self.lm_loss_weight}, video_loss_weight: {self.video_loss_weight}, \
                info_loss_weight: {self.info_loss_weight}, ref_loss_weight: {self.ref_loss_weight}, \
                    uncertainty_loss_weight: {self.uncertainty_loss_weight}, and \
                        tv_loss_weight: {self.tv_loss_weight} for training")


    def get_model(self):
        return self.model

    def connector(self, frames):
        return self.get_model().mm_projector(frames)

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def vision_encode(self, vision_tower, frames):
        frame_features = vision_tower(frames)
        return frame_features

    def post_projector_pooling(self, image_feature):
        stride = self.config.video_pooling_stride
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim).contiguous()
        return image_feature

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        informative_labels: Optional[torch.LongTensor] = None,
        relevance_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        frames: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        model_outputs = copy.copy(outputs)
        hidden_states = outputs[0]
        outputs = outputs[1:]
        logits = self.lm_head(hidden_states).float()




        if self.config.video_head_stop_grad:
            hidden_states_no_grad = hidden_states.detach()
        else:
            hidden_states_no_grad = hidden_states
        
        informative_logits = self.informative_head(hidden_states_no_grad).float()
        relevance_logits = self.relevance_head(hidden_states_no_grad).float()
        # relevance_logits = torch.sigmoid(relevance_logits)
        if not self.training:
            relevance_logits = torch.clamp(relevance_logits, 0, 1)
        log_variance = self.uncertainty_head(hidden_states_no_grad).float()
        variance = torch.exp(torch.clamp(log_variance, min=-6, max=2))

        # NOTE: all labels used here are already shifted in data collator
        ce_loss_fct = CrossEntropyLoss(ignore_index=-100)
        mse_loss_fct = MSELoss()
        loss = 0.

        if labels is not None:
            if not(labels != -100).any():
                labels[:, 0] = input_ids[:, 1]      # make sure lm_loss is calculated for every example, or the deepspeed training process will hang
            lm_loss = ce_loss_fct(logits.flatten(0, 1), labels.flatten())
            if not return_dict:
                outputs = (logits,) + outputs + (loss,)
        else:
            lm_loss = 0.
        

        info_loss = 0
        ref_loss = 0
        uncertainty_loss = 0
        tv_loss = 0

        # Informative head: CE loss for classification 
        if informative_labels is not None:

            if not (informative_labels != -100).any():
                informative_labels[:, 0] = 0  # Ensure valid target for at least one example
            info_loss = ce_loss_fct(
                informative_logits.flatten(0, 1), 
                informative_labels.flatten(0, 1)
            )
        # Relevance head: MSE + uncertainty (NLL) loss
        if relevance_labels is not None:

            if not (relevance_labels != -100).any():
                relevance_labels[:, 0] = 0  # make sure video_loss is calculated for every example, or the deepspeed training process will hang
            
            valid_mask = (relevance_labels != -100)
            # we only enforce tv loss if there is more than one point

            relevance_logits = relevance_logits.squeeze(-1) # [B, T, 1] -> [B, T]
            if relevance_logits.shape[1] > 1:
                # Total variation loss to enforce smoothness
                tv_mask = valid_mask[:, 1:]
                tv_mask.mul(valid_mask[:, :-1]) # Binary 1 and 0 of which are valid
                tv_loss = torch.mean((relevance_logits[:, 1:] - relevance_logits[:, :-1]) ** 2)
                tv_loss = (tv_mask * tv_loss).sum() / (tv_mask.sum() + 1e-6)


            relevance_logits_flat = relevance_logits.flatten().float()
            relevance_labels_flat = relevance_labels.flatten().float()
            valid_mask = valid_mask.flatten()


            relevance_logits_valid = relevance_logits_flat[valid_mask]
            relevance_labels_valid = relevance_labels_flat[valid_mask]


            if relevance_labels_valid.numel() > 1:
                mse_loss_fct = MSELoss()
                ref_loss = mse_loss_fct(
                    relevance_logits_valid,
                    relevance_labels_valid
                )
            else:
                ref_loss = torch.tensor(0.0, device=relevance_logits.device)


            min_log_var = math.log(1 / (2 * math.pi))  # ≈ -1.8379
            max_log_var = 2.0
            log_variance_clamped = torch.clamp(log_variance, min=min_log_var, max=max_log_var)
            variance = torch.exp(log_variance_clamped)

            residual = relevance_labels_valid - relevance_logits_valid
            # Note: flatten variance and log_variance_clamped similarly
            variance_valid = variance.flatten(0, 1)[valid_mask]
            # Gaussian NLL loss
            nll_loss = (residual ** 2) / (2 * variance_valid + 1e-6)  + 0.5 * torch.log(2 * math.pi * variance_valid)
            uncertainty_loss = nll_loss.mean()
            uncertainty_loss = torch.clamp(uncertainty_loss, min=0)




        ref_loss_with_smoothness = ref_loss + self.tv_loss_weight * tv_loss 

        video_loss = self.info_loss_weight * info_loss + self.ref_loss_weight * ref_loss_with_smoothness + self.uncertainty_loss_weight * uncertainty_loss

        loss = lm_loss * self.lm_loss_weight + video_loss * self.video_loss_weight
        

        

        if int(os.environ['RANK']) == 0 and self.training:
            loss_logs = {
                "train/tv_loss": tv_loss.item() if tv_loss != 0 else None,
                "train/lm_loss": lm_loss.item() if lm_loss != 0 else None,
                "train/info_loss": info_loss.item() if info_loss != 0 else None,
                "train/ref_loss": ref_loss.item() if ref_loss != 0 else None,
                "train/uncertainty_loss": uncertainty_loss.item() if uncertainty_loss != 0 else None,
                "train/video_loss": video_loss.item() if video_loss != 0 else None,
                "train/total_loss": loss.item(), #if not isinstance(loss, float) else loss,
            }

            weighted_logs = {
                "train/tv_loss": tv_loss.item()*self.tv_loss_weight if tv_loss != 0 else None,
                "train/lm_loss": lm_loss.item()*self.lm_loss_weight if lm_loss != 0 else None,
                "train/info_loss": info_loss.item()*self.info_loss_weight if info_loss != 0 else None,
                "train/ref_loss": ref_loss.item()*self.ref_loss_weight if ref_loss != 0 else None,
                "train/uncertainty_loss": uncertainty_loss.item()*self.uncertainty_loss_weight if uncertainty_loss != 0 else None,
                "train/video_loss": video_loss.item()*self.video_loss_weight if video_loss != 0 else None,
                "train/total_loss": loss.item()
            }
            print(f" Mean pred relevance: {relevance_logits_valid.mean().item():.4f}")
            print(weighted_logs)

            loss_logs = {k: v for k, v in weighted_logs.items() if v is not None}
            wandb.log(loss_logs)
        



        if not return_dict:
            outputs = (loss,) + outputs
            return outputs

        vid_head_results = VideoHeadCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            lm_loss=lm_loss,
            video_loss=video_loss,
            informative_logits=informative_logits,
            relevance_logits=relevance_logits,
            uncertainty=log_variance # uncertainty
        )

        return vid_head_results

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        frames: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        '''
        The original generate function of LLaVA.
        '''
        logger.warning('You are calling the generate function of LLaVA, which is deprecated for Live Video models. Please use a LiveInfer class for inference.')
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().generate(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        return outputs


def build_video_head_live_llava_qwen(**kwargs):
    model, tokenizer = build_live(config_class=VideoHeadLiveLlavaQwenConfig, model_class=VideoHeadLiveLlavaQwenForCausalLM, **kwargs)
    for param in model.get_vision_tower().parameters():
        param.requires_grad = False
    return model, tokenizer

if __name__ == '__main__':
    from transformers import HfArgumentParser
    from models.arguments_live import LiveTrainingArguments
    args, = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
    args.llm_pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    print(args.to_dict())
    model, tokenizer = build_video_head_live_llava_qwen(is_training=True, **args.to_dict())
    print(model.config, tokenizer)
    for name, param in model.named_parameters():
        if param.numel() == 0:
            print(f"⚠️ Param with 0 elements: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")

