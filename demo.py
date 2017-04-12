
import os
import sys
import tensorflow as tf
import numpy as np

import config as cfg
#path = cfg.config_path()
#sys.path.append(path.top_dir)

import inception

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
    FLAGS.batch_size = param.batch_size
    FLAGS.num_gpus = param.gpu_num
    FLAGS.run_once = param.test_run_once

    return  0
def test():
    test_init()
    print("test function now!")
    inception.camelyon_test()

    return 0

def main():
    cfg_run = cfg.config_run()
    cfg_const = cfg.config_const()
    print("main")
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
    main()





