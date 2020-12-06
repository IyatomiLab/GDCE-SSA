#!/bin/sh

cd src

# CAE
# =================================

# vanilla
python train_classification.py \
    --dataset livedoor \
    --char_len 80 \
    --train_batch 256 \
    --val_batch 256 \
    --test_batch 256 \
    --character_encoder CAE \
    --encode_dim 10 \
    --num_workers 4

# +WT
python train_classification.py \
    --dataset livedoor \
    --char_len 80 \
    --train_batch 256 \
    --val_batch 256 \
    --test_batch 256 \
    --character_encoder CAE \
    --encode_dim 10 \
    --da wt \
    --num_workers 4

# +SSA
python train_classification.py \
    --dataset livedoor \
    --char_len 80 \
    --train_batch 256 \
    --val_batch 256 \
    --test_batch 256 \
    --character_encoder CAE \
    --encode_dim 10 \
    --da ssa \
    --gamma 484 \
    --num_workers 4

# =================================

# BetaVAE
# =================================

# vanilla
python train_classification.py \
    --dataset livedoor \
    --char_len 80 \
    --train_batch 256 \
    --val_batch 256 \
    --test_batch 256 \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --encode_dim 10 \
    --num_workers 4

# +WT
python train_classification.py \
    --dataset livedoor \
    --char_len 80 \
    --train_batch 256 \
    --val_batch 256 \
    --test_batch 256 \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --encode_dim 10 \
    --da wt \
    --num_workers 4

# +SSA
python train_classification.py \
    --dataset livedoor \
    --char_len 80 \
    --train_batch 256 \
    --val_batch 256 \
    --test_batch 256 \
    --character_encoder BetaVAE \
    --beta 8.0 \
    --encode_dim 10 \
    --da ssa \
    --gamma 2.0 \
    --num_workers 4

# =================================
