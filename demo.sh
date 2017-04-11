#!/bin/sh

DIR_DATA_TOP="/home1/zhuzj/dataset/camelyon16_B2"
#DIR_TRAIN="${DIR_DATA_TOP}/raw-data/train/"
#DIR_EVAL="${DIR_DATA_TOP}/raw-data/validation/"
DIR_DATA="${DIR_DATA_TOP}/TFRecords_all/"
DIR_TEST_DATA="${DIR_DATA_TOP}/test_data/"
DIR_CSV_OUTS="${DIR_DATA_TOP}/csv_outs/"
DIR_EVAL_MODEL="${DIR_DATA_TOP}/eval_models/"
DIR_TEST_MODEL="${DIR_DATA_TOP}/test_models/"
DIR_TRAIN_MODEL="${DIR_DATA_TOP}/models_bak_0408/"
PATH_MODEL_CHECKPOINT="${DIR_DATA_TOP}/models_bak_0408/model.ckpt-04082045-80000"
DIR_LOG="${DIR_DATA_TOP}/logs/"

batch_size=128
num_gpu=2
init_learning_rate=0.01
max_steps=160000
num_examples=8196
test_examples_cnt=2370610

input_model="$1"
#echo "$input_model"

touch WORKSPACE

if [ "${input_model}" = "train" ];then
    echo "train mode ...."
    echo "Build The Model"
    bazel build inception/camelyon_train

    #rm -rf ${DIR_TRAIN_MODEL}
    mkdir -p ${DIR_TRAIN_MODEL}
    mkdir -p ${DIR_LOG}
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

elif [ "${input_model}" = "eval" ];then
    echo "evaluate  mode ..."

    bazel build inception/camelyon_eval

    bazel-bin/inception/camelyon_eval \
        --eval_dir="${DIR_EVAL_MODEL}" \
        --data_dir="${DIR_DATA}" \
        --subset=validation \
        --checkpoint_dir="${DIR_TRAIN_MODEL}"\
        --input_queue_memory_factor=1 \
        --num_examples="${num_examples}" \
        --run_once

elif [ "${input_model}" = "test" ];then
    bazel build inception/camelyon_test

    mkdir -p ${DIR_TEST_MODEL}
    mkdir -p ${DIR_CSV_OUTS}
    bazel-bin/inception/camelyon_test \
        --test_dir="${DIR_TEST_MODEL}" \
        --data_dir="${DIR_DATA}" \
        --subset=test \
        --csv_dir="${DIR_CSV_OUTS}" \
        --checkpoint_dir="${DIR_TRAIN_MODEL}" \
        --input_queue_memory_factor=1 \
        --num_examples="${test_examples_cnt}" \
        --batch_size="${batch_size}" \
        --num_gpus="${num_gpu}" \
        --run_once

else
    echo "input param fail!"

fi


