import sys
import os.path as osp
import numpy as np

sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), './')))

import camelyon_train
import camelyon_eval
import camelyon_data
