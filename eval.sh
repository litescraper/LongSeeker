#!/bin/bash
    
# Skip already evaluated items (set to "true" to skip, "false" to re-evaluate)
SKIP_EXISTING="true"

# Dataset path configuration
DATASET_PATH="dataset/browsecomp.json"

# Result directory configuration
RESULT_DIR="result/browsecomp_meta_react_separate"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

python -u eval.py \
    --result_dir ${RESULT_DIR} \
    --dataset_path ${DATASET_PATH} \
    --num_workers 10 \
    ${SKIP_EXISTING:+--skip_existing ${SKIP_EXISTING}}