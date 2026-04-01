#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ETTm2" ]; then
    mkdir ./logs/ETTm2
fi

dataset=ETTm2
seq_len=336
feature_dim=7
K=8
n_clusters=6
n_prototypes=6
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
        >logs/ETTm2/${dataset}_${seq_len}_${pred_len}.log
done
