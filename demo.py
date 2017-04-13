from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import config as cfg

import inception.camelyon_test as camelyon_test
import inception.camelyon_train as camelyon_train
import inception.camelyon_eval as camelyon_eval

FLAGS = tf.app.flags.FLAGS

def train_init():
    path = cfg.config_path()
    inception_path = os.path.join(path.top_dir,path.inception_dir)
    sys.path.append(inception_path)


def train():
    train_init()


def eval_init():
    return 0

def eval():
    eval_init()

    return 0

def test_init():
    path = cfg.config_path()
    run = cfg.config_run()
    param = cfg.config_param()
    csv_path = os.path.join(path.data_dir_top,path.csv_dir)
    if os.path.exists(csv_path):
        print("%s is exist!"%(csv_path))
    else:
        os.makedirs(csv_path)
        print("creat the dir %s!" %(csv_path))

    FLAGS.test_dir = str(os.path.join(path.data_dir_top,path.test_dir))
    FLAGS.data_dir = str(os.path.join(path.data_dir_top,path.data_dir))
    FLAGS.csv_dir = str(os.path.join(path.data_dir_top,path.csv_dir))
    FLAGS.checkpoint_dir = str(os.path.join(path.data_dir_top,path.mode_dir))
    FLAGS.subset=str(run.mode)
    FLAGS.input_queue_memory_factor = param.input_queue_memory_factor
    FLAGS.num_examples = param.test_num_example
    FLAGS.batch_size = param.test_batch_size
    FLAGS.num_gpus = param.gpu_num
    FLAGS.run_once = param.test_run_once

    return  0

def test():
    test_init()
    print("test function now!")
    camelyon_test.main()
    return 0


def print_mode():
    cfg_run = cfg.config_run()
    print("================================================")
    print("===                                          ===")
    print("===              %s mode                   ==="%  (cfg_run.mode))
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





