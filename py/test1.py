import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

from common.layers import Affine, Relu
import numpy as np

reLu = Relu()
#reLu.forward(1)
reLu.forward(np.array([-1.0, 1.0, 2.0]))
reLu.backward(np.array([-1.0, 1.0, 2.0]))

# Affine 全连接层：y = xW + b
W = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
b = np.array([0.1, 0.2, 0.3])
x = np.array([[2.0, 3.0]])

affine = Affine(W, b)
y = affine.forward(x)
print("forward:", y)

# dout 是损失对 y 的梯度；backward 返回 dx，并保存 dW、db
dout = np.ones_like(y)
dx = affine.backward(dout)
print("dx:", dx)
print("dW:", affine.dW)
print("db:", affine.db)

#x= np.array([[1.0, 2.0],
#forward: [[ 9.1 12.2 15.3]]
#dx: [[ 6. 15.]]
#dW: [[1. 1. 1.]
# [2. 2. 2.]]
#db: [1. 1. 1.]

#x = np.array([[2.0, 3.0]])
#forward: [[14.1 19.2 24.3]]
#dx: [[ 6. 15.]]
#dW: [[2. 2. 2.]
# [3. 3. 3.]]
#db: [1. 1. 1.]

#x = np.array([[1.0, 2.0],
#              [3.0, 4.0]])
#forward: [[ 9.1 12.2 15.3]
# [19.1 26.2 33.3]]
#dx: [[ 6. 15.]
# [ 6. 15.]]
#dW: [[4. 4. 4.]
# [6. 6. 6.]]
#db: [2. 2. 2.]