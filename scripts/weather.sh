#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Weather" ]; then
    mkdir ./logs/Weather
fi

dataset=weather
seq_len=336
feature_dim=21
K=10
n_clusters=8
n_prototypes=8
epochs=50
batch_size=256
pretrain_epochs=30
pretrain_lr=0.001
pretrain_batch_size=256

for pred_len in 96 192; do
    python -u scripts/run_full_pipeline.py \
        --dataset $dataset \
        --seq_len $seq_len \
        --pre_len $pred_len \
        --feature_dim $feature_dim \
        --K $K \
        --n_clusters $n_clusters \
        --n_prototypes $n_prototypes \
        --epochs $epochs \
        --batch_size $batch_size \
        --pretrain_epochs $pretrain_epochs \
        --pretrain_lr $pretrain_lr \
        --pretrain_batch_size $pretrain_batch_size \
        >logs/Weather/${dataset}_${seq_len}_${pred_len}.log
done
