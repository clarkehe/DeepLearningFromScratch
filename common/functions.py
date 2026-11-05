# coding: utf-8
import numpy as np

#激活函数

# 恒等函数
def identity_function(x):
    return x

# 阶跃函数：
# 0: x <= 0,
# 1: x > 0
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# S 型函数：用于逻辑回归
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

# S 型函数的导数（梯度）
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    
# 修正线性单元
def relu(x):
    return np.maximum(0, x)

# 修正线性单元导数（梯度）
def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

# Softmax 函数（归一化指数函数）是将一个任意实数向量压缩并归一化为概率分布向量的数学工具.
# 用于分类
def softmax(x):
    if x.ndim == 2:
        # 这段 softmax 转置，是为了按列计算：
        # 假设输入形状是 (批次数, 类别数)，转置后每一列对应一个样本，
        # 于是用 axis=0 对类别做归一化，最后再转置回来。
        x = x.T
        maxes = np.max(x, axis=0) # 每列找最大值
        x = x - maxes
        sums = np.sum(np.exp(x), axis=0) # 每列求和
        expX = np.exp(x)
        # expX 是10行3列，sums是1行3列
        # expX / sums 是怎么除的呢？
        # 这是广播机制，sums 会被广播成10行3列，每一列都是相同的值，然后再进行逐元素相除。
        # expX 中每一列除的值是一样的，都是该列的和。这样就实现了每列归一化。
        y = expX / sums # 每列归一化
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# 均方误差(MSE)
# y:预测数据
# t:监督数据
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交叉熵损失函数
# 可能有多组数据，最后除batch_size，求平均。
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)# 按行找最大值索引
             
    batch_size = y.shape[0] #多少行数据。 每一行，取出正确解标签对应的预测值，一行是一张图片的预测值
    # 交叉熵损失函数公式：-Σt*log(y)
    # 取出正确解标签对应的预测值，求对数，再求和
    # 1e-7是为了避免log(0)的情况
    # y[np.arange(batch_size), t] 取出每一行中正确解标签对应的预测值
    # np.log(tmp + 1e-7) 求对数，再求和，最后除以batch_size，求平均。
    # 交叉�熵损失函数的公式是：-Σt*log(y)，其中t是正确解标签，y是预测值。
    # 对于one-hot编码的t，只有正确解标签对应的y会被取出，其他的t都是0，所以只需要取出正确解标签对应的y即可。
    tmp = y[np.arange(batch_size), t]
    return -np.sum(np.log(tmp + 1e-7)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
