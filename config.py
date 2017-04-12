#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

class config_run():
    def __init__(self):
        self.mode="test"

class config_path:
    def __init__(self):
        self.data_dir_top="/home1/zhuzj/dataset/camelyon16_B2/"
        self.data_dir = "TFRecords_ALL/"
        self.train_dir="events/train/"
        self.eva_dir="events/eval/"
        self.test_dir="events/test/"
        self.csv_dir="csv_outs"
        self.mode_dir="models_bak_0408"
        self.model_checkpoint="model.ckpt-04082045-80000"
        self.top_dir= sys.path[0]
        self.inception_dir ="inception"

class config_param:
    def __init__(self):
        self.init_learning_rate=0.01
        self.input_queue_memory_factor=1
        self.max_step = 160000
        self.gpu_num = 2
        self.batch_size=64
        self.fine_tune =1
        self.eval_num_example=8196
        self.test_num_example=2370610
        self.test_run_once = 1
        self.test_batch_size = 64

class config_const:
    def __init__(self):
        self.TRAIN_MODE="train"
        self.EVAL_MODE="eval"
        self.TEST_MODE="test"

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