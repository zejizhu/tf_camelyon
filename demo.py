import os
import sys
import config as cfg
import numpy as np
path = cfg.config_path()
sys.path.append(path.top_dir)
import inception

def train_init():
    path = cfg.config_path()
    inception_path = os.path.join(path.top_dir,path.inception_dir)
    sys.path.append(inception_path)


def train():
    train_init()


def eval_init():
    return 0

def eval(a):
    b= np.log()
    return 0

def test_init():
    return  0
def test():
    return 0

def main():
    cfg_run = cfg.conifg_run()
    if cfg.init() != 0:
        print("config init fail!")
        return 1



if __name__ == "main":
    main()




