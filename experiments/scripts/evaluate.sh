#!/usr/bin/env bash

GPU_ID=$1
MODEL=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py \
    --id 'sgmn' \
    --elimination True \
    --batch_size 64 \
    --model_method 'sgmn' \
    --evaluate True \
    --model ${MODEL}
