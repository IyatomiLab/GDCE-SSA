#!/bin/sh

cd src

# CAE
python train_character_embedding.py \
    --character_encoder CAE \
    --encode_dim 10 \
    --train_batch 64 \
    --weight_decay 0 \
    --num_epoch 1000

# BetaVAE
python train_character_embedding.py \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --encode_dim 10 \
    --train_batch 64 \
    --weight_decay 0
