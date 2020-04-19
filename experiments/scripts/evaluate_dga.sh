#!/usr/bin/env bash

GPU_ID=$1
MODEL=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train_dga.py \
    --id 'dga' \
    --batch_size 64 \
    --model_method 'dga' \
    --evaluate True \
    --model ${MODEL}
