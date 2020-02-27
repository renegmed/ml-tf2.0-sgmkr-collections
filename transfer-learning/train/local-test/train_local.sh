#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

docker run -v $(pwd)/test_dir:/opt/ml \
        -e SM_NUM_GPUS=0 \
	-e SM_TRAIN_STEPS=100 \
        -e SM_MODEL_DIR=/opt/ml/model \
        -e SM_DATA_DIR=/opt/ml/input/data/training \
        --rm ${image} train

