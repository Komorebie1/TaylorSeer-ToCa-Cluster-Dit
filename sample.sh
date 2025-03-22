#!/bin/bash

base_command="python sample.py \
    --image-size 256 \
    --num-sampling-steps 50 \
    --interval 4 \
    --max-order 4 \
    --ratio-scheduler ToCa-ddim50 \
    --soft-fresh-weight 0.25 \
    --ddim-sample \
    --cluster-method kmeans \
    --cluster-nums 16 \
    --smooth-rate 0.007 \
    --topk 1 \
    "


eval $base_command --fresh-ratio 0.0