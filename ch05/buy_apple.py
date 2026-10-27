# coding: utf-8
from layer_naive import *


apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1

# 税前总价每增加 1 元，税后价格约增加 1.1 元；
# 税率系数每增加 1，税后价格增加 200 元。
# 这些是变化的敏感度.
dapple_price, dtax = mul_tax_layer.backward(dprice)

# 苹果单价每增加 1 (1.1) 元，税前总价增加 2 (2.2)元；
# 苹果数量每增加 1 (1.1) 个，税前总价增加 100 (110) 元。
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)
