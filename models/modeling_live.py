import torch, os
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, Cache
from transformers.utils import logging
from peft import prepare_model_for_kbit_training
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from transformers import BitsAndBytesConfig
from bitsandbytes.nn import Linear4bit
from deepspeed import zero

from .tokenization_live import build_live_tokenizer_and_update_config
from .vision_live import build_live_vision

logger = logging.get_logger(__name__)

class LiveMixin(AutoModelForCausalLM):
    def set_vision_inside(self):
        logger.warning_once("!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction.")
        if not hasattr(self, 'vision_encoder'):
            self.vision_encoder, self.vision_encode = build_live_vision(self.config)
        else:
            logger.warning_once("Vision encoder already exists, skip setting vision encoder inside the model.")

    def unset_vision_inside(self):
        del self.vision_encoder
        del self.vision_encode

    def visual_embed(self, frames: torch.Tensor):
        if hasattr(self, 'vision_encode'):
            frames = self.vision_encode(self.vision_encoder, frames)
        frames = self.connector(frames)
        if hasattr(self, 'post_projector_pooling'):
            frames = self.post_projector_pooling(frames)
        return frames.view(-1, frames.shape[-1])

    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
    ):
        if frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            return self.visual_embed(frames)
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            # inputs_embeds[v_mask] = self.visual_embed(frames).to(inputs_embeds.dtype)
            visual_embeds = self.visual_embed(frames).to(inputs_embeds.dtype)  # [N, D]
            B, S, D = inputs_embeds.shape
            inputs_embeds_flat = inputs_embeds.view(-1, D)
            v_mask_flat = (input_ids == self.config.v_placeholder_id).view(-1)  # [B * S]
            inputs_embeds_flat_updated = inputs_embeds_flat.clone()
            inputs_embeds_flat_updated[v_mask_flat] = visual_embeds
            inputs_embeds = inputs_embeds_flat_updated.view(B, S, D)


        return inputs_embeds


def fast_greedy_generate(*, model: LiveMixin, inputs_embeds: torch.Tensor, past_key_values: Cache, eos_token_id: int, inplace_output_ids: torch.Tensor,
                         repetition_penalty=None, generated_token_ids=list()):
    if repetition_penalty is not None:
        assert isinstance(repetition_penalty, float)
        logits_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

    for i in range(inplace_output_ids.size(1)):
        outputs = model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        if repetition_penalty is not None:
            if len(generated_token_ids) > 0:
                outputs_logits = logits_processor(
                    input_ids=torch.tensor(generated_token_ids).unsqueeze(0).to(device=inplace_output_ids.device, dtype=torch.long), scores=outputs.logits[:, -1, :])
                outputs_logits = outputs_logits.unsqueeze(1)
            else:
                outputs_logits = outputs.logits[:, -1:]
            new_token_id = outputs_logits.argmax(dim=-1)
            if not new_token_id == eos_token_id:        # special tokens should not be penalized
                generated_token_ids.append(new_token_id.item())
        else:
            outputs_logits = outputs.logits
            new_token_id = outputs_logits[:, -1:].argmax(dim=-1)
        inplace_output_ids[:, i] = new_token_id
        if new_token_id == eos_token_id:
            break
        inputs_embeds = model.get_input_embeddings()(new_token_id)
    return inplace_output_ids[:, :i+1], past_key_values, generated_token_ids





def build_live(
    *,
    is_training: bool,
    config_class: type,
    model_class: type,
    llm_pretrained: str = None,
    lora_pretrained: str = None,
    finetune_modules: list[str] = None,
    lora_modules: str = None,
    lora_r: int = None,
    lora_alpha: int = None,
    set_vision_inside: bool = False,
    attn_implementation: str = 'flash_attention_2',
    torch_dtype: str | torch.dtype = 'auto',
    quantization: bool = False,
    **kwargs
):

    if quantization:
        logger.info("Quantization applied")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
            llm_int8_skip_modules=finetune_modules
        )
    
        model = model_class.from_pretrained(
            llm_pretrained, 
            config=config_class.from_pretrained(llm_pretrained, **kwargs),
            torch_dtype=torch_dtype, 
            attn_implementation=attn_implementation,
            device_map='cuda' if torch.cuda.device_count() == 1 or dist.is_initialized() else 'auto',
            quantization_config=quantization_config
            )
        
        model = prepare_model_for_kbit_training(model)
        

    else:
        model = model_class.from_pretrained(
            llm_pretrained, 
            # low_cpu_mem_usage=False,
            config=config_class.from_pretrained(llm_pretrained, **kwargs),
            torch_dtype=torch_dtype, 
            attn_implementation=attn_implementation,
            device_map='cuda' if torch.cuda.device_count() == 1 or dist.is_initialized() else 'auto',
            )
                        
    tokenizer = build_live_tokenizer_and_update_config(llm_pretrained, model.config)
    logger.warning(f"model config after update: {model.config}")
    if is_training:
        if lora_pretrained:
            print(f'loading lora from checkpoint: {lora_pretrained}')
            model = PeftModel.from_pretrained(model, lora_pretrained, is_trainable=True) # we are further fine-tuning
        else:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_modules,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                modules_to_save=finetune_modules,
                inference_mode=False,
            )
            print(f'creating lora with config: {lora_config}')
            
            model = get_peft_model(model, lora_config)
        
        

        model.print_trainable_parameters()

    else:
        if lora_pretrained:
            logger.info(f'loading lora from checkpoint: {lora_pretrained}')
            model = PeftModel.from_pretrained(model, lora_pretrained, is_trainable=False)
        else:
            logger.warning(f'!!! Fail to load lora from checkpoint: {lora_pretrained}. Return a new initialized model.')
        if set_vision_inside:
            model.set_vision_inside()
        model.requires_grad_(False)
        model.currently_training = False

    return model, tokenizer




            # print("In the correct if else bracket")

            # old_peft_model = "wangyueqian/MMDuet"
            # model.relevance_head = nn.Linear(3584, 2, bias=False)
            # old_model = PeftModel.from_pretrained(model, old_peft_model)
            # old_sd = old_model.state_dict()
            # for name, param in old_model.named_parameters():
            #     print(name, param.shape, param.dtype)
            
            # print(f"Informative_head weight")
            # print(old_model.informative_head.weight)
            # print(f"Uncertainty_head weight")
            # print(old_model.uncertainty_head.weight)
            # print("loaded MMDuet")


            # lora_config = LoraConfig(
            #     r=lora_r,
            #     lora_alpha=lora_alpha,
            #     target_modules=lora_modules,
            #     lora_dropout=0.05,
            #     task_type="CAUSAL_LM",
            #     modules_to_save=finetune_modules,
            #     inference_mode=True,
            # )
            # print(f'Loading in new model: {lora_config}')
            # model.relevance_head = nn.Linear(3584, 1, bias=False)
            # model = get_peft_model(model, lora_config)
            # print("Loaded new model")


            # new_sd = model.state_dict()
            # filtered_sd = {}

            # print("Starting transfer")
            # for k, v in old_sd.items():
            #     # Check for relevance head mismatch
            #     if "relevance_head" in k:
            #         if k in new_sd and v.shape != new_sd[k].shape:
            #             print(f"Skipping (mismatch): {k}")
            #             continue
            #     if k in new_sd:
            #         print(k, v)
            #         filtered_sd[k] = v
            # missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
            # for name, param in model.named_parameters():
            #     print(name, param.shape, param.dtype)
            # # Handles both PEFT-wrapped and non-wrapped
            # def get_actual_layer(layer):
            #     return layer.original_module if hasattr(layer, "original_module") else layer

            # get_actual_layer(model.lm_head).weight.data = get_actual_layer(old_model.lm_head).weight.data.clone()
            # get_actual_layer(model.informative_head).weight.data = get_actual_layer(old_model.informative_head).weight.data.clone()
            # print(get_actual_layer(old_model.relevance_head).weight.data.shape, get_actual_layer(model.relevance_head).weight.data.shape)
            # get_actual_layer(model.relevance_head).weight.data = get_actual_layer(old_model.relevance_head).weight.data.clone()[0:1, :]

            # print("✅ Missing keys:", missing)
            # print("✅ Unexpected keys:", unexpected)
            # # === Step 4 (optional): Copy over partial weights for relevance_head ===

            # # Example: copy first row of old weight matrix
            # if "relevance_head.weight" in old_sd:
            #     old_w = old_sd["relevance_head.weight"]
            #     if model.relevance_head.original_module.weight.shape[0] == 1 and old_w.shape[0] == 2:
            #         model.relevance_head.original_module.weight.data = old_w[0:1, :]
            #         print("🔁 Copied over first row of relevance_head.weight")

            # if "relevance_head.bias" in old_sd and model.relevance_head.bias is not None:
            #     old_b = old_sd["relevance_head.bias"]
            #     if model.relevance_head.bias.shape[0] == 1 and old_b.shape[0] == 2:
            #         model.relevance_head.bias.data = old_b[0:1]
            #         print("🔁 Copied over first entry of relevance_head.bias")
            

            


            # assert torch.equal(model.model.model.layers[0].self_attn.k_proj.lora_A.default.weight.data, old_model.model.model.layers[0].self_attn.k_proj.lora_A.default.weight.data)
            # assert torch.equal(get_actual_layer(model.lm_head).weight.data , get_actual_layer(old_model.lm_head).weight.data )
            # assert torch.equal(get_actual_layer(model.informative_head).weight.data , get_actual_layer(old_model.informative_head).weight.data)
            # assert torch.equal(get_actual_layer(model.relevance_head).weight.data , get_actual_layer(old_model.relevance_head).weight.data)
            # # === Step 5: Save the LoRA + heads checkpoint ===

            # save_dir = "old_weights/merged_weights"
            # model.save_pretrained(save_dir, safe_serialization=True)
            # tokenizer.save_pretrained(save_dir)
            # print(f"✅ Saved modified LoRA model to {save_dir}")

            # exit(0)