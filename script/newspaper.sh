#!/bin/sh

cd src

# CAE
# =================================

# vanilla
python train_classification.py \
    --dataset newspaper \
    --char_len 128 \
    --character_encoder CAE \
    --encode_dim 10 \
    --train_batch 512 \
    --val_batch 512 \
    --test_batch 400 \
    --num_workers 4

# +WT
python train_classification.py \
    --dataset newspaper \
    --char_len 128 \
    --character_encoder CAE \
    --encode_dim 10 \
    --train_batch 512 \
    --val_batch 512 \
    --test_batch 400 \
    --da wt \
    --num_workers 4

# +SSA
python train_classification.py \
    --dataset newspaper \
    --char_len 128 \
    --character_encoder CAE \
    --encode_dim 10 \
    --train_batch 512 \
    --val_batch 512 \
    --test_batch 400 \
    --da ssa \
    --gamma 363.0 \
    --num_workers 4

# =================================

# BetaVAE
# =================================

# vanilla
python train_classification.py \
    --dataset newspaper \
    --char_len 128 \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --encode_dim 10 \
    --train_batch 512 \
    --val_batch 512 \
    --test_batch 400 \
    --num_workers 4

# +WT
python train_classification.py \
    --dataset newspaper \
    --char_len 128 \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --encode_dim 10 \
    --train_batch 512 \
    --val_batch 512 \
    --test_batch 400 \
    --da wt \
    --num_workers 4

# +SSA
python train_classification.py \
    --dataset newspaper \
    --char_len 128 \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --encode_dim 10 \
    --train_batch 512 \
    --val_batch 512 \
    --test_batch 400 \
    --da ssa \
    --gamma 1.5 \
    --num_workers 4

# =================================
