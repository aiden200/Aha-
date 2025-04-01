# Aha! – When Has a Model Seen Enough? Adaptive Video Segmentation and Highlight Detection

<div align="center">
    <img src="assets/cover_photo.jpg">
    <p></p>
</div>


<!-- Official implementation of paper *VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format* -->

# Introduction

Aha! is a video-language model designed to recognize when enough information has been observed in a video, triggering reflection and segmentation at the right moments, just like human intuition during "Aha!" moments.

Unlike traditional video-language models that process continuously or respond at fixed intervals, Aha! dynamically determines when to pause, analyze, and extract key moments based on contextual importance.

By fine-tuning Qwen-7B with an importance-aware segmentation mechanism and integrating uncertainty-based decision-making, Aha! intelligently:
- Segments video streams when enough information is available
- Ranks and extracts key moments using task-aware importance scoring
- Generates highlight reels with structured summarization techniques

This approach enables more efficient video understanding, making Aha! applicable to autonomous agents, surveillance, video summarization, and decision-support systems.


# Installation
1. Create conda environment and use pip to install some packages
```shell
git clone --recurse-submodules https://github.com/aiden200/Aha-.git
cd aha

conda create -n aha python=3.10
conda activate aha
pip install --upgrade pip
pip install -r requirements.txt
```


2. Install torch compiled with cuda. Install them together using the instructions provided by [pytorch.org](https://pytorch.org).


3. Install llava. If you run into any issues check the [official repository download instructions.](https://github.com/LLaVA-VL/LLaVA-NeXT)
```bash
cd LLaVA_NeXT
pip install -e ".[train]"
cd ..
```

4. Install flash-attention following the instructions in [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). If you have difficulties installing it, add `--attn_implementation sdpa` in every command to use the sdpa implementation of transformer attention for train or inference.
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir 
```

<!-- 5. Download MMDuet checkpoints from HuggingFace: [https://huggingface.co/wangyueqian/MMDuet](https://huggingface.co/wangyueqian/MMDuet) and put the files under folder `./outputs/mmduet`.

```bash
mkdir outputs
cd outputs
git clone https://huggingface.co/wangyueqian/MMDuet mmduet
cd ..
``` -->


## Common Problems

*Note 1:* If you get a `bitsandbytes` error, try running:
```bash
pip uninstall bitsandbytes
pip install bitsandbytes
```
*Note 2:* If you get a `Undefined symbol cpython-310-x86_64-linux-gnu.so: undefined symbol:` error, try running:
```
pip uninstall flash-attn
pip install flash-attn --no-build-isolation --no-cache-dir
```
*Note 3:* If you get some kind of `c10 deprecation` error, your pytorch version might be too high. The authors used the version:
- `Python==3.10`
- `torch == 2.5.1`
- `torchvision==0.20.1`
- `cuda12.4`

```bash
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu124
```

*Note 4:* If you want to use the CPU adam optimizer with deepspeed, you need to install it with the correct flags:
```bash
DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
pip install deepspeed \
  --global-option="build_ext" \
  --global-option="-j8"
```



# Inference
## Download Our model
<!-- 
## Download pretrained Model
- Download checkpoints from HuggingFace: (https://huggingface.co/aiden200/Aha-) and put the files under folder `./outputs/aha`
```bash
mkdir outputs
cd outputs
git clone https://huggingface.co/aiden200/Aha- aha
cd ..
``` -->

# Evaluation

## Download the tvsum dataset
Follow the instructions from the official [tvsum](https://github.com/yalesong/tvsum?tab=readme-ov-file) repository then move it to the datasets folder as `datasets/tvsum`

Run `scripts/inference/tvsum.sh`.

# Training
<!-- 
## Download pretrained MMDuet Model
- Download MMDuet checkpoints from HuggingFace: (https://huggingface.co/wangyueqian/MMDuet) and put the files under folder `./outputs/mmduet`
```bash
mkdir outputs
cd outputs
git clone https://huggingface.co/wangyueqian/MMDuet mmduet
cd ..
``` -->


## Prepare the annotations
Download the datasets folder which contains the metadata for our dataset. You can download them from our [huggingface page](https://huggingface.co/datasets/aiden200/aha-annotationsv1/tree/main). 

Use the following commands:
```
git lfs install
git clone https://huggingface.co/datasets/aiden200/aha-annotationsv1
mv aha-annotationsv1/datasets .
rm -rf aha-annotationsv1
```

This should give you a structure like this.


```
├── datasets
│   ├── charades
│   │   └── annotations
│   │       └── test-random_prompt.json
│   ├── coin
│   │   └── annotations
│   │       └── train-0.25_0.5_earlier-120s_240s.json
│   ├── download_tools
│   │   ├── coin_download.py
│   │   ├── coin_files.json
│   │   ├── hisum_download.py
│   │   ├── mr_hisum_crawler.py
│   │   ├── mr_hisum_metadata.csv
│   │   └── vocabulary.csv
│   ├── hisum
│   │   └── annotations
│   │       ├── mr_hisum_metadata.csv
│   │       └── split.json
│   ├── qvh
│   │   └── annotations
│   │       ├── highlight_val-random_prompt.json
│   │       └── highlight_val_release.jsonl
│   ├── shot2story
│   │   └── annotations
│   │       ├── dvc_train-human_anno-0.25_0.5_earlier.json
│   │       ├── magqa_test.json
│   │       └── magqa_train-0.25_0.5-earlier.json
│   ├── tvsum
│   └── youcook2
│       └── annotations
│           └── val-random_prompt.json
├── assets
├── configs
├── data
├── demo
├── instructions
├── LICENSE
├── LLaVA_NeXT
├── models
├── README.md
├── requirements.txt
├── scripts
├── test
├── train.py
└── Utils
```


## Prepare the data

- Mr.HiSum:
  - Prepare the `mr_hisum.h5` file following the instructions of the [official repo](https://github.com/MRHiSum/MR.HiSum).
  - Place the `mr_hisum.h5` file in the `datasets/hisum/annotations` folder.
  <!-- - Run the metadata generator -->

- Download the videos, and link each video folder to `datasets/${DATASET_NAME}/videos`. Recommended download links for each dataset:
  - Mr.HiSum: [https://github.com/MRHiSum/MR.HiSum](https://github.com/MRHiSum/MR.HiSum).
  - Shot2Story: [https://huggingface.co/mhan/shot2story-videos](https://huggingface.co/mhan/shot2story-videos)
  - COIN: [https://coin-dataset.github.io/](https://coin-dataset.github.io/)



Since some of these datasets (expecially Mr.HiSum) are very big, you can always specify the video path in the `configs/datasets/ahait.json` file where the datasets exist on your local machine.  

*note:* I've left a script to help the download processes at `datasets/download_tools`

When running training code for the first time, the dataset code will traverse all videos of the training dataset and measure the frame rate, duration, number of frames, and the corruption status of the videos. It will store this information in `datasets/${dataset_name}/videos_metadata.json`. This can take some time, since some of these datasets are very large.

- Download [paraphrase-en.gz](https://github.com/lichengunc/refer/raw/refs/heads/master/evaluation/meteor/data/paraphrase-en.gz) (59MB) which is used for dense video captioning evaluation. Put this file at `test/dvc/metrics/data/paraphrase-en.gz`



## Run the training script
Log into wandb in order to monitor your progress
```bash
wandb login [YOUR_API_KEY]
```

Log into huggingface in order to save your new model weights 
```bash
huggingface-cli login
```
and update `models.arguments_live`
```python
push_to_hub=True,
hub_model_id=[REPO NAME],
```

Start the training process
```bash
bash ./scripts/train.sh
```

## Distributed Training
This model is very big, trained on 8 V100 GPUs, and you will probably need to utilize distributed training. [I've included instructions on how to train on the cloud, using Paperspace.](instructions/distributed_instructions.md)

# Acknowledgments
This work was conducted as part of the author's AEOP Fellowship, with compute resources and mentorship provided by the Army Research Laboratory, West Coast (ARL-W).

Our codebase is built on top of the VideoLLM-Online and MMDuet frameworks. The base model is derived from Qwen, which itself is an adaptation of LLaVA-NeXT:
- [VideoLLM-online](https://github.com/showlab/VideoLLM-online)
- [MMDuet](https://github.com/yellow-binary-tree/MMDuet)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) 


<!-- 
## Inference and evaluation
Scripts to inference on all benchmarks are listed in `./scripts/inference/`.

**WARNING**: Each script file contains many steps for inference and evaluation. DO NOT directly run these script files. Instead, read the contents of these files carefully and run them step by step.

- YouCook2 dense video captioning: `./scripts/inference/youcook2.sh`
- Shot2Story-MAGQA-39k multi-answer grounded video question answering (MAGQA): `./scripts/inference/magqa.sh`
  - **Note**: To save compute, we do not calculate the similarity score between the pred answer and the gold answer if the pred time is not in the gold timespan. We simply set this score to 1 in the score matrix of evaluator_output. These scores are not used in calculating and do not affect the final metric (in-span score).
- Charades-STA temporal video grounding: `./scripts/inference/charades.sh`
- QVHighlights highlight detection: `./scripts/inference/qvh.sh`


# Training

- If you want to reproduce the training process, you also need to download the training data. Download the videos, and link each video folder to `datasets/${DATASET_NAME}/videos`. Here we list recommended video download links, while you can also download from other sources:
  - COIN: [https://huggingface.co/datasets/WHB139426/coin](https://huggingface.co/datasets/WHB139426/coin)
  - HiREST: [https://github.com/j-min/HiREST](https://github.com/j-min/HiREST)
  - DiDeMo: [https://github.com/LisaAnne/TemporalLanguageRelease](https://github.com/LisaAnne/TemporalLanguageRelease)
  - QueryD: [https://www.robots.ox.ac.uk/~vgg/data/queryd/](https://www.robots.ox.ac.uk/~vgg/data/queryd/)

Run `./scripts/train.sh`.

When running training code for the first time, the dataset code will traverse all videos of the training dataset and stat the frame rate, duration and number of frames of the videos, and store this information in `datasets/${dataset_name}/videos_metadata.json`. This can take quite a long time.
Considering that videos downloaded from different sources may be slightly different, in order to ensure that the videos are correctly loaded, we do not include this metadata information in our data release.
``` -->

# Citation
If you find this work useful in your research, please consider citing:
```bibtex
@misc{chang2025aha,
      title={Aha! – When Has a Model Seen Enough? Adaptive Video Segmentation and Highlight Detection},
      author={Aiden Chang and Stephanie Lukin and Celso De Melo},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/},
}
