# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#データ点、観測値
X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.79, -0.89, -0.04])

# 定数項＋ガウス基底(12次元) 
def phi(x): 
    s = 0.2 # ガウス基底の「幅」
    return np.append(1, np.exp(-(x - np.arange(0, 1 + s, s)) ** 2 / (2 * s * s)))

#ハイパーパラメータ
alpha = 0.1 
beta = 9.0 

#計画行列
PHI = np.array([phi(x) for x in X])

#線形回帰
w = np.linalg.solve(np.dot(PHI.T, PHI), np.dot(PHI.T, t))

#ベイズ線形回帰
Sigma_N = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
mu_N = beta * np.dot(Sigma_N, np.dot(PHI.T, t))


# 正規分布の確率密度関数
def normal_dist_pdf(x, mean, var): 
    return np.exp(-(x-mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

# 2次形式( x^T A x を計算)
def quad_form(A, x):
    return np.dot(x, np.dot(A, x))

xlist = np.arange(0, 1.0, 0.01)
tlist = np.arange(-1.5, 1.5, 0.01)
z = np.array([normal_dist_pdf(tlist, np.dot(mu_N, phi(x)),
        1 / beta + quad_form(Sigma_N, phi(x))) for x in xlist]).T
plt.contourf(xlist, tlist, z, 5, cmap=plt.cm.binary)
plt.plot(xlist, [np.dot(mu_N, phi(x)) for x in xlist], 'r')
plt.plot(xlist, [np.dot(w, phi(x)) for x in xlist], 'b') 
plt.plot(X, t, 'go')
plt.show()
