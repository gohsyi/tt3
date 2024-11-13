#!/bin/bash

huggingface-cli login --token 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

git clone https://github.com/gohsyi/tt3.git
cd tt3

conda env create -f environment.yaml
conda activate yy

ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port $PORT1 z_sa_3.py \
    --model_name google/gemma-2-9b-it \
    --train_set_path openai/gsm8k \
    --deepspeed ./deepspeed_configs/deepspeed_3.json\
    --max_length 256 \
    --save_every_steps 50 \
    --per_device_train_batch_size 4 \

huggingface-cli upload Q_models/tt_3/saved_model gohsyi/gemma-2-9b-it-em --token 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
