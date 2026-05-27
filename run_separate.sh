#!/bin/bash

# unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
USE_META_TOOLS=${USE_META_TOOLS:-true}

# Dataset path configuration
DATASET_PATH="dataset/browsecomp.json"

python -u main_separate.py \
    --tool_count_max 300 \
    --dataset ${DATASET_PATH} \
    --num_workers 30 \
    --use_meta_tools ${USE_META_TOOLS} \