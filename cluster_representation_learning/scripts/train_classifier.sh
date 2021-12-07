#!/bin/bash

python classifier_trainer.py  \
    net.conv1_kernel_size=3 \
    net.model_n_out=5 \
    misc.num_gpus=8 \
    misc.out_dir=${OUT_DIR} \
    net.pretrained_weights=$PRETRAIN \
    net.finetuned_weights=$FINETUNE \
