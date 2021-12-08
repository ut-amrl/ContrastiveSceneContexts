#!/bin/bash

python classification_results_generator.py  \
    net.conv1_kernel_size=3 \
    net.model_n_out=5 \
    misc.num_gpus=1 \
    misc.out_dir=${OUT_DIR} \
    net.pretrained_weights=$PRETRAIN \
    data.results_out=${OUT_FILE} \
