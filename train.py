import json
from dataclasses import asdict

import torch
from models import build_model_and_tokenizer, parse_args
from data import (
    build_concat_train_dataset_from_config, get_data_collator
)
from transformers import Trainer
from deepspeed import zero
import wandb
import yaml
import os
from dotenv import load_dotenv


load_dotenv()

class TrainerWithLossErrorCatch(Trainer):
    def training_step(self, model, inputs):
        try:
            loss = super().training_step(model, inputs)
            if int(os.environ['RANK']) == 0:
                wandb.log({"train_loss": loss})
            return loss
        except Exception as e:
            print(f"Error during training step: {e}, use a dummy loss = 0.0")
            return torch.tensor(0., device=self.args.device,
                                dtype=torch.float16 if self.args.fp16 else torch.bfloat16 if self.args.bf16 else torch.float32)  # dummy loss

    def log(self, logs):
        print(f"[Rank {os.environ['RANK']}] Trainer log: {logs}")
        super().log(logs)


def safe_print_model_params(model, local_rank, global_rank):
    if global_rank == 0:  # or use `local_rank == 0 and global_rank == 0`
        with zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
            for name, param in model.named_parameters():
                print(f"{name}, shape={param.shape}, dtype={param.dtype}, requires_grad={param.requires_grad}")



def rank0_print(*args, local_rank, global_rank):
    if local_rank == 0 and global_rank == 0:
        print(*args)




def train_model(args, local_rank, global_rank):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exists_ok=True)

    device = torch.device("cuda")
    print(f"GPU {local_rank} - Using device: {device}")

    with open("configs/wandb/wandb.config", 'r') as f:
        wandb_config = yaml.safe_load(f)
        
    # print(args)

    run=None
    if global_rank == 0:
        wandb.init(
            entity=wandb_config['wandb']['entity'],
            project=wandb_config['wandb']['project'],
            config=wandb_config['wandb']['config']
        )
        print("Wandb initialized")

    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))

    if 'llava' in args.llm_pretrained:
        image_processor = model.get_vision_tower().image_processor
    else:
        image_processor = None
    

    

    # Deepspeed might have trouble if we access the params after partitioning
    for name, param in model.named_parameters():
        rank0_print((name, param.shape, param.dtype, param.requires_grad), local_rank=local_rank, global_rank=global_rank)

    print(f"[Rank {global_rank}] Distributed initialized? {torch.distributed.is_initialized()}")
    print(f"[Rank {global_rank}] Backend: {torch.distributed.get_backend()}")

    # We load the datasets.
    train_dataset_config = json.load(open(args.dataset_config))


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
    
    # trainer = Trainer(
    #     model=model, tokenizer=tokenizer,
    #     args=args,
    #     train_dataset=train_dataset,
    #     data_collator=data_collator,
    #     # callbacks=[FreezeVITCallBack()]
    # )

    print("Starting Training!")
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Finished Training")
    if global_rank == 0:
        trainer.save_model()
        if args.push_to_hub:
            print("Saving model to huggingface")
            trainer.push_to_hub()


    if run is not None:
        run.finish()
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()



def train():
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    assert local_rank != -1, "LOCAL_RANK environment variable not set"
    assert global_rank != -1, "RANK environment variable not set"
    print(f"Global rank {global_rank}, Local Rank: {local_rank} initiated")
    args = parse_args('train')
    train_model(args, local_rank, global_rank)

if __name__ == "__main__":
    train()
