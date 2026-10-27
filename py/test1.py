import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

from common.layers import Relu
import numpy as np

reLu = Relu()
#reLu.forward(1)
reLu.forward(np.array([-1.0, 1.0, 2.0]))
reLu.backward(np.array([-1.0, 1.0, 2.0]))
