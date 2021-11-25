#!/bin/bash

python cluster_rep_trainer.py  \
    net.conv1_kernel_size=3 \
    net.model_n_out=64 \
    misc.num_gpus=1 \
    misc.out_dir=${OUT_DIR} \
    net.pretrained_weights=$PRETRAIN \
    net.finetuned_weights=$FINETUNE \
