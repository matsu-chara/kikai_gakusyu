# -*- coding:utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt

#φ(x)基底関数 多項式 
def phi(x,m):
	dim_list = range(0, m)
	return [ x**dim for dim in dim_list]

#φ(x)基底関数 ガウス基底 
def phi2(x, m):
	s = 0.1
	temp = range(0, m)
	mu_list = [float(t)/m for t in temp]

	return [ math.exp( - (x-mu)**2 / (2*(s**2)) ) for mu in mu_list ]

	
#回帰式
def f(w, phi):
	return np.dot(w, phi)


# X:データ点, t:観測値	
X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.89, -0.79, -0.04])

M = 8	      #基底の次元
I = np.identity(M) #単位行列
phi_list = [phi(x,M) for x in X] #データを基底関数に突っ込む
PHI = np.array(phi_list) #基底関数Φ(x_n)をデータごとに並べ立てたもの

lam_list = [0, 0.001,0.01]
xlist = np.arange(0, 1, 0.01)
phi_list = [phi(x,M) for x in xlist]

for lam in lam_list:
	w = np.linalg.solve( (lam*I + np.dot(PHI.T, PHI)), np.dot(PHI.T, t)) # w=(λI+Φ^t・Φ)^-1・Φ^t・tを計算

	#出力
	ylist = [f(w, phi) for phi in phi_list]
	plt.plot(xlist, ylist)

plt.plot(X, t, 'o')
plt.show()
