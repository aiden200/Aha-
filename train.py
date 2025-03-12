import json
from dataclasses import asdict

import torch
from models import build_model_and_tokenizer, parse_args
from data import (
    build_concat_train_dataset_from_config, get_data_collator
)
from transformers import Trainer
import wandb
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

class TrainerWithLossErrorCatch(Trainer):
    def training_step(self, model, inputs):
        # try:
        loss = super().training_step(model, inputs)
        # if wandb and torch.distributed.get_rank() == 0:
        #     wandb.log({"loss": loss})
        return loss
        # We don't want to use this for now
        # except Exception as e:
        #     print(f"Error during training step: {e}, use a dummy loss = 0.0")
        #     return torch.tensor(0., device=self.args.device,
        #                         dtype=torch.float16 if self.args.fp16 else torch.bfloat16 if self.args.bf16 else torch.float32)  # dummy loss


def rank0_print(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)


def train():

    # print(torch.distributed.is_initialized())
    # exit(0)

    with open("configs/wandb/wandb.config", 'r') as f:
        wandb_config = yaml.safe_load(f)
    
    
    
    args = parse_args('train')
    if torch.distributed.get_rank() == 0:
        run = wandb.init(
            entity=wandb_config['wandb']['entity'],
            project=wandb_config['wandb']['project'],
            config=wandb_config['wandb']['config']
        )
    # print(torch.distributed.is_initialized())
    rank0_print(args)
    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    
    if 'llava' in args.llm_pretrained:
        image_processor = model.get_vision_tower().image_processor
    else:
        image_processor = None

    for name, param in model.named_parameters():
        rank0_print(name, param.shape, param.dtype, param.requires_grad)


    # We load the datasets. In our case, its only tvsum. No need to change anything
    train_dataset_config = json.load(open(args.dataset_config))

    # Need to correct this
    train_dataset = build_concat_train_dataset_from_config(
        tokenizer=tokenizer, config=train_dataset_config
    )


    data_collator = get_data_collator(tokenizer=tokenizer, image_processor=image_processor, model_config=model.config, **asdict(args))

    args.gradient_checkpointing_kwargs = {'use_reentrant': False}

    trainer = TrainerWithLossErrorCatch(
        model=model, tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    with torch.cuda.amp.autocast():
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    run.finish()

if __name__ == "__main__":
    train()
