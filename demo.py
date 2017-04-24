#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import config as cfg
from inception import  *

#import inception.camelyon_test as camelyon_test
#import inception.camelyon_train as camelyon_train
#import inception.camelyon_eval as camelyon_eval

FLAGS = tf.app.flags.FLAGS

def train_init():
    path = cfg.config_path()
    run = cfg.config_run()
    param = cfg.config_param()
    save_models_path = os.path.join(path.data_dir_top,path.save_models_dir)
    if os.path.exists(save_models_path):
        print("save models dir exist!")
    else:
        print("creat save models dir!")
        os.makedirs(save_models_path)

    FLAGS.train_dir = str(save_models_path)
    FLAGS.data_dir = str(os.path.join(path.data_dir_top,path.data_dir))
    FLAGS.fine_tune = param.fine_tune
    FLAGS.initial_learning_rate = param.init_learning_rate
    FLAGS.input_queue_memory_factor = param.input_queue_memory_factor
    FLAGS.checkpoint_dir = str(os.path.join(path.data_dir_top, path.models_dir))
    FLAGS.method_name = str(run.method_name)
    #FLAGS.pretrained_model_checkpoint_path = str(os.path.join(path.data_dir_top,path.models_dir,path.model_checkpoint))
    FLAGS.batch_size = param.train_batch_size
    FLAGS.num_gpus =param.gpu_num
    FLAGS.max_steps = param.max_step
    FLAGS.subset = str(run.mode)

def train():
    train_init()
    camelyon_train.main()
    return 0

def eval_init():
    path = cfg.config_path()
    #run = cfg.config_run()
    param = cfg.config_param()
    eavl_param = cfg.param_eval()

    FLAGS.num_gpus = param.gpu_num

    FLAGS.eval_dir = str(os.path.join(path.data_dir_top,path.event_top_dir,eavl_param.event_dir))
    FLAGS.data_dir = str(os.path.join(path.data_dir_top,eavl_param.tfrecord_dir))

    FLAGS.checkpoint_dir = str(os.path.join(path.data_dir_top, eavl_param.model_dir))
    FLAGS.input_queue_memory_factor = eavl_param.input_queue_memory_factor
    FLAGS.num_examples = eavl_param.num_example
    FLAGS.batch_size = eavl_param.batch_size
    FLAGS.subset = eavl_param.mode_name
    FLAGS.run_once = eavl_param.run_once
    return 0

def eval():
    eval_init()
    camelyon_eval.main()
    return 0

def test_init():
    path = cfg.config_path()
    run = cfg.config_run()
    param = cfg.config_param()
    test_param = cfg.param_test()
    csv_path = os.path.join(path.data_dir_top,test_param.csv_outs)
    if os.path.exists(csv_path):
        print("%s is exist!"%(csv_path))
    else:
        os.makedirs(csv_path)
        print("creat the dir %s!" %(csv_path))

    event_path =os.path.join(path.data_dir_top, path.event_top_dir,test_param.event_dir)
    if os.path.exists(event_path):
        print("%s is exist!"%(event_path))
    else:
        os.makedirs(event_path)
        print("creat the dir %s!" %(event_path))

    FLAGS.subset=str(run.mode)
    ## new param mode
    FLAGS.test_dir = str(event_path)
    FLAGS.checkpoint_dir = str(os.path.join(path.data_dir_top, test_param.model_dir))
    FLAGS.data_dir = str(os.path.join(path.data_dir_top, test_param.tfrecord_dir))
    FLAGS.csv_dir = csv_path
    FLAGS.input_queue_memory_factor = test_param.input_queue_memory_factor
    FLAGS.num_examples = test_param.num_example
    FLAGS.batch_size = test_param.batch_size
    FLAGS.run_once = test_param.run_once
    FLAGS.test_gpu_id = test_param.gpu_id

    return  0

def test():
    test_init()
    camelyon_test.main()
    return 0


def print_mode():
    cfg_run = cfg.config_run()
    print("================================================")
    print("===                                          ===")
    print("                 %5s mode                   "%  (cfg_run.mode))
    print("===                                          ===")
    print("================================================")

def main(unused_argv=None):
    print_mode()
    cfg_run = cfg.config_run()
    cfg_const = cfg.config_const()
    if cfg.init() != 0:
        print("config init fail!")
        return 1
    if cfg_run.mode == cfg_const.TRAIN_MODE:
        train()
    elif cfg_run.mode == cfg_const.EVAL_MODE:
        eval()
    elif cfg_run.mode == cfg_const.TEST_MODE:
        test()
    else:
        print("Input run mode[%s] fail!" %(cfg_run.mode))
    return 0

if __name__ == "__main__":
    tf.app.run()





