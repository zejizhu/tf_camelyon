import os
import sys
import config as cfg
path = cfg.config_path()
sys.path.append(path.top_dir)
import inception

def train_init():
    path = cfg.config_path()
    inception_path = os.path.join(path.top_dir,path.inception_dir)
    sys.path.append(inception_path)


def train():
    train_init()
    path = cfg.config_path()
    param = cfg.config_param()
    build_tf ="bazel build inception/camelyon_train"
    print(build_tf)
    os.system(build_tf)
    TRAIN_DIR =os.path.join(path.data_dir_top,path.train_dir)
    DATA_DIR=os.path.join(path.data_dir_top,path.data_dir)
    EVA_DIR=os.path.join(path.data_dir_top,path.eva_dir)
    run_train="bazel-bin/inception/camelyon_train --train_dir=%s --data_dir=%s --eval_dir=%s --fine_tune=%s " \
              "--initial_learning_rate=%lf --input_queue_memory_factor=%d" \
              %(TRAIN_DIR,DATA_DIR,EVA_DIR,param.fine_tune,param.init_learning_rate,param.input_queue_memory_factor)
    print(run_train)
    os.system(run_train)

def eval():
    return 0


def main():
    print("train next!")
    train()




if __name__ == "main":
    main()




