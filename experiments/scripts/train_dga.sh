#!/usr/bin/env bash

GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train_dga.py \
    --learning_rate 1e-4 \
    --id 'dga' \
    --word_drop_out 0.2 \
    --rnn_drop_out 0.2 \
    --jemb_drop_out 0.2 \
    --batch_size 64 \
    --edge_gate_drop_out 0.0 \
    --word_judge_drop 0.0 \
    --model_method 'dga'
