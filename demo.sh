#!/usr/bin/env bash

DIR_DATA_TOP="/home1/zhuzj/dataset/camelyon16_B2"
DIR_TRAIN="${DIR_DATA_TOP}/raw-data/train/"
DIR_EVAL="${DIR_DATA_TOP}/raw-data/validation/"
DIR_DATA="${DIR_DATA_TOP}/TFRecords/"
DIR_MODEL="${DIR_DATA_TOP}/models/"

num_gpu=1
init_learning_rate=0.001


touch WORKSPACE
echo "Build The Model"
bazel build inception/camelyon_train

rm -rf ${DIR_MODEL}
mkdir -p ${DIR_MODEL}

bazel-bin/inception/camleyon_train \
    --train_dir="${DIR_TRAIN}" \
    --data_dir="${DIR_DATA}" \
    --pretrained_model_checkpoint_path="${DIR_MODEL}" \
    --fine_tune=True \
    --initial_learning_rate="${init_learning_rate}" \
    --input_queue_memory_factor=1 \
    --num_gpus="${num_gpu}"

