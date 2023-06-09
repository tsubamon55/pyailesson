import numpy as np
import matplotlib.pylab as plt

x = np.array([1, 3, 5, 7])
# 上層i番目からの出力

w = np.array([1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1],
             )
# 上層j番目出力と下層k番目入力の間の重み
#
#     　下層
#  　　ーーーーーー
#  上　|
#  層　|
#  　　|

b = np.array([1, 1, 1, 1])
#  (下層の)バイアス

u = np.dot(x, w) + b
#  x,w,bより求めた下層の仮出力たち

