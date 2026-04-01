#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Electricity" ]; then
    mkdir ./logs/Electricity
fi

dataset=electricity
seq_len=336
feature_dim=321
K=16
n_clusters=12
n_prototypes=12
epochs=50
batch_size=128
pretrain_epochs=30
pretrain_lr=0.001
pretrain_batch_size=128

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
        >logs/Electricity/${dataset}_${seq_len}_${pred_len}.log
done
