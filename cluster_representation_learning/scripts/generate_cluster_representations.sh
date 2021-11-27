#!/bin/bash

python cluster_rep_generator.py  \
    net.conv1_kernel_size=3 \
    net.model_n_out=64 \
    misc.num_gpus=1 \
    net.pretrained_weights=$PRETRAIN \
    data.results_out=${OUT_FILE}
