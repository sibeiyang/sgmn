#!/usr/bin/env bash

GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py \
    --id 'sgmn' \
    --learning_rate 1e-4 \
    --word_drop_out 0.2 \
    --rnn_drop_out 0.2 \
    --jemb_drop_out 0.2 \
    --elimination True \
    --batch_size 64 \
    --model_method 'sgmn'
