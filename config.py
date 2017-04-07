import os
import sys

class config_path:
    def __init__(self):
        self.data_dir_top="/home1/zhuzj/dataset/camelyon16_B2/"
        self.train_dir="raw-data/train"
        self.data_dir="TFRecords/"
        self.eva_dir="raw-data/validation"
        self.model_checkpoint_path="model"
        self.top_dir= sys.path[0]
        self.inception_dir ="inception"

class config_param:
    def __init__(self):
        self.init_learning_rate=0.001
        self.input_queue_memory_factor=1
        self.max_step = 10000
        self.gpu_cnt = 2
        self.fine_tune =1






def config_init():
    path = config_path()
    DATA_DIR_TOP = path.data_dir_top
    if os.path.exists(DATA_DIR_TOP):
        print("%s is exists!" %(DATA_DIR_TOP))
    else:
        print("%s is not exists !" %(DATA_DIR_TOP))
        return 1
    tmp_path = os.path.join(DATA_DIR_TOP,path.model_checkpoint_path)
    if os.path.exists(tmp_path):
        print("%s is exists!" % (tmp_path))
    else:
        os.makedirs(tmp_path)
    return  0
