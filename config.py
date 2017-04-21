#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

class config_run():
    def __init__(self):
        self.mode="test"

class config_path:
    def __init__(self):
        self.data_dir_top="/home1/zhuzj/dataset/camelyon16_B2"
        self.data_dir = "TFRecords_HSD2/"
        self.test_data_dir = "TFRecords_HSD2/"
        self.train_dir = "events/train/"
        self.eva_dir = "events/eval/"
        self.test_dir = "events/test/"
        self.csv_dir = "csv_outs_hsd2"
        self.event_top_dir = "events"
        self.models_dir = "models_hsd2_batch_128_bak"
        self.model_checkpoint = "model.ckpt-04202236-110000"
        self.top_dir = sys.path[0]
        self.inception_dir = "inception"

class config_param:
    def __init__(self):
        self.init_learning_rate = 0.01
        self.input_queue_memory_factor = 1
        self.max_step = 120000
        self.gpu_num = 2
        self.train_batch_size = 128
        self.fine_tune =0
        self.eval_num_example=5952
        self.eval_batch_size = 32
        self.test_num_example=2370610
        self.test_run_once = 1
        self.test_batch_size = 128

## TRAIN PARAM ##
class train_param:
    def __init__(self):
        self.init_learning_rate = 0.01
        self.input_queue_memory_factor = 1
        self.max_step = 120000
        self.gpu_num = 2
        self.batch_size = 64
        self.fine_tune =1
        self.event_dir  = "train_event"
        self.tfrecord_dir = "TFRecords_ALL"

## EVAL PARAM ##
class eval_param:
    def __init__(self):
        self.input_queue_memory_factor = 1
        self.num_example = 8192
        self.run_once = 1
        self.batch_size = 128
        self.gpu_num = 2
        self.event_dir = "eval_event"
        self.tfrecord_dir = "TFRecords_ALL"

## TEST PARAM ##
class test_param:
    def __init__(self):
        self.input_queue_memory_factor = 1
        self.num_example = 2370610
        self.run_once = 1
        self.batch_size = 128
        self.gpu_num = 2
        self.event_dir = "test_event"
        self.csv_outs = "csv_outs"
        self.tfrecord_dir = "TFRecords_ALL"

class config_const:
    def __init__(self):
        self.TRAIN_MODE = "train"
        self.EVAL_MODE = "validation"
        self.TEST_MODE = "test"

def dir_check():
    path = config_path()
    train_dir=os.path.join(path.data_dir_top,path.data_dir)
    if os.path.exists(train_dir):
        print("%s is exists!" %(train_dir))
    else:
        os.makedirs(train_dir)
        print("creat the dit %s! "%(train_dir))
    eval_dir = os.path.join(path.data_dir_top,path.eva_dir)
    if os.path.exists(eval_dir):
        print("%s is exists!" %(eval_dir))
    else:
        os.makedirs(eval_dir)
        print("creat the dit %s! "%(eval_dir))

    test_dir = os.path.join(path.data_dir_top,path.test_dir)
    if os.path.exists(test_dir):
        print("%s is exists!" %(test_dir))
    else:
        os.makedirs(test_dir)
        print("creat the dit %s! "%(test_dir))

    return 0
def mode_check():
    run = config_run()
    const_val = config_const()
    if run.mode == const_val.TEST_MODE:
        print("the run mode is %s!" %(run.mode ))
    elif run.mode == const_val.TRAIN_MODE:
        print("the run mode is %s!" % (run.mode))
    elif run.mode == const_val.EVAL_MODE:
        print("the run mode is %s!" % (run.mode))
    else:
        print("Input mode [%s] fail! "%(run.mode))
        return  1
    return  0

def init():
    if mode_check()!=0:
        return 1
    if dir_check() !=0:
        return 1
    return  0


if __name__ == "__main__":
    print("run config init!")
    init()
