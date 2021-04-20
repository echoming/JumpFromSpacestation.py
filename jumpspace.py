import math
from math import exp
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

import numpy as np

import matplotlib.pyplot as plt

print("hello")


# 计算密度
def density_cal(h):
    rho0 = 1.225
    T0 = 288.15
    if h <= 11000:
        T = T0 - 0.0065 * h
        Rho = rho0 * ((T / T0) ** 4.25588)
    elif 11000 < h <= 20000:
        T = 216.65
        Rho = 0.36392 * exp((-h + 11000) / 6341.62)
    else:
        T = 216.65 + 0.001 * (h - 20000)
        Rho = 0.088035 * ((T / 216.65) ** -35.1632)
    return Rho


# 计算重力加速度
def acc_gravity_cal(h):
    G = 6.67259 * (10 ** -11)
    M = 5.965 * (10 ** 24)
    R = 6371393
    acc = G * M / (R + h) / (R + h)
    return acc


# 空气阻力计算
def air_resistance_cal(v, density_t):
    S = np.pi * 5 * 5
    C = 0.415
    f = 0.5 * C * density_t * S * v * v
    return f


x = np.arange(0, 300000, 1)
# print(x)
density = []
gravity = []
for i in x:
    y_1 = density_cal(i)
    y_2 = acc_gravity_cal(i)
    density.append(y_1)
    gravity.append(y_2)
# print(density[1])
# 有了 x 和 y 数据之后，我们通过 plt.plot(x, y) 来画出图形，并通过 plt.show() 来显示。
plt.plot(x, density)
plt.show()

plt.plot(x, gravity)
plt.show()

vector = [0]
a = []
for i in x:
    v = vector[i] * vector[i] + 2 * (
            acc_gravity_cal(300000 - i) - air_resistance_cal(vector[i], density_cal(300000 - i)) / 65)
    v = v ** 0.5
    vector.append(v)

del vector[0]
plt.plot(300000 - x, vector)
plt.show()

print('Max Speed：%.30f' % max(vector))
print('End Speed:%.30f' % vector[300000 - 1])

