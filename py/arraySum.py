# coding: utf-8
import numpy as np

# 示例矩阵有 3 行、3 列：
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

sumx = np.sum(x)         # 不指定 axis：所有元素相加，结果是 45
sum_columns = np.sum(x, axis=0)  # 沿第 0 维求和：逐列相加，结果是 [12, 15, 18]
sum_rows = np.sum(x, axis=1)     # 沿第 1 维求和：逐行相加，结果是 [6, 15, 24]

print("所有元素求和：", sumx)
print("按列求和（axis=0）：", sum_columns)
print("按行求和（axis=1）：", sum_rows)

# reshape(x.shape[0], -1) 示例：
# 3 个样本，每个样本是一个 2×3 的矩阵，整体形状为 (3, 2, 3)
x_batch = np.array([[[1, 2, 3],
                     [4, 5, 6]],

                    [[7, 8, 9],
                     [10, 11, 12]],
                     
                    [[13, 14, 15],
                     [16, 17, 18]]])

# 保留第 0 维的 3 个样本；将每个样本剩余的 2×3 展平为 6 个数
x_flat = x_batch.reshape(x_batch.shape[0], -1)

print("reshape 前的形状：", x_batch.shape)
print("reshape 后的形状：", x_flat.shape)
print("reshape 后的数组：\n", x_flat)

# 一维数组的随机生成示例：
# 生成一个包含 2 个随机数的一维数组，范围在 [0, 1) 之间
#随机生成的数组 X： [0.50789892 0.34528555]
# 可以理解为一行，两列的矩阵，形状为 (2,)
X = np.random.rand(2)
print("随机生成的数组 X：", X)

# 二维数组的随机生成示例：
# 生成一个 2 行 3 列的随机数组，范围在 [0, 1) 之间
#随机生成的权重矩阵 W：
# [[0.24387856 0.49923112 0.65399033]
# [0.80258535 0.23255024 0.25758101]]
W = np.random.rand(2,3)
print("随机生成的权重矩阵 W：\n", W)