#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

docker run -v $(pwd)/test_dir:/opt/ml \
        -e epochs=2 \
        -e batch-size=1024 \
        -e SM_NUM_GPUS=0 \
        -e SM_MODEL_DIR=/opt/ml/model \
        -e SM_INPUT_FILE=/opt/ml/input/data/ratings.csv \
        --rm ${image} train

