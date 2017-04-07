#!/bin/sh

DIR_DATA_TOP="/home1/zhuzj/dataset/camelyon16_B2"
#DIR_TRAIN="${DIR_DATA_TOP}/raw-data/train/"
#DIR_EVAL="${DIR_DATA_TOP}/raw-data/validation/"
DIR_DATA="${DIR_DATA_TOP}/TFRecords/"
DIR_EVAL_MODEL="${DIR_DATA_TOP}/eval_models/"
DIR_TRAIN_MODEL="${DIR_DATA_TOP}/models/"
PATH_MODEL_CHECKPOINT="${DIR_DATA_TOP}/models/model.ckpt-5000"


batch_size=64
num_gpu=2
init_learning_rate=0.003
max_steps=5000

input_model="${1%/}"

touch WORKSPACE

if [ "${input_model} = "train" ];then
    echo "train mode ...."
    echo "Build The Model"
    bazel build inception/camelyon_train

    #rm -rf ${DIR_TRAIN_MODEL}
    mkdir -p ${DIR_TRAIN_MODEL}

    bazel-bin/inception/camelyon_train \
        --train_dir="${DIR_TRAIN_MODEL}" \
        --data_dir="${DIR_DATA}" \
        --fine_tune=False \
        --initial_learning_rate="${init_learning_rate}" \
        --input_queue_memory_factor=1 \
        --pretrained_model_checkpoint_path= "${PATH_MODEL_CHECKPOINT}"\
        --batch_size="${batch_size}" \
        --num_gpus="${num_gpu}" \
        --max_steps="${max_steps}"

elif [ ${input_model} = "eval" ];then
    echo "evaluate  mode ..."

    bazel build inception/camelyon_eval

    bazel-bin/inception/camelyon_eval \
        --eval_dir="${DIR_EVAL_MODEL}" \
        --data_dir="${DIR_DATA}" \
        --subset=validation \
        --num_examples=500 \
        --checkpoint_dir="${DIR_TRAIN_MODEL}"\
        --input_queue_memory_factor=1 \
        --run_once


else
    echo "input param fail!"

fi