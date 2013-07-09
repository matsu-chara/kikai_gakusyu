# -*- coding:utf-8 -*-
import math
import numpy as np
import cvxopt as co
import cvxopt.solvers
import matplotlib.pyplot as plt
import support_vector_machine_hard as hsvm

#2クラス判別SVM ソフトマージン（ハードとの差分のみ）
class softSVM(hsvm.hardSVM):

    #C is a parameter which controls balance between penalty and margin
    def __init__(self, C):
        self.C = C

    def QuadraticProgramming(self, N, x, t):
        #objective function (1/2.)x^tQ^tx - a^tx
        temp = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                temp[i, j] = t[i] * t[j] * self.kernel(x[i], x[j])

        Q =  co.matrix(temp)
        p =  co.matrix(-np.ones(N))

        #constraint Gx <= h
        preG = np.vstack(( -np.identity(N), np.identity(N) ))
        G = co.matrix(preG)
        preh = np.hstack(( np.zeros(N), C*np.ones(N) ))
        h = co.matrix(preh)

        # constraint Ax=b
        A = co.matrix(t, (1,N))
        b = co.matrix(0.0)

        sol = cvxopt.solvers.qp(Q, p, G, h, A, b)
        a = np.array(sol['x']).reshape(N)
        return a

if __name__ == '__main__':
    N = 200
    C = 1e+1

    mu1    = [1, 0]
    mu2    = [1, 2]
    sigma = [[1.0, 0.3], [0.3, 1.0]]

    data1 = np.random.multivariate_normal(mu1, sigma, N/2)
    data2 = np.random.multivariate_normal(mu2, sigma, N/2)

    svm = softSVM(C)
    svm.train(data1, data2)

    #クラス判別
    # y = svm.classify( np.array([-2, -4]) )
    # if(y>=0):
    #     print('class1', y)
    # else:
    #     print('class2', y)

    # 訓練データを描画
    x1, x2 = data1.transpose()
    plt.plot(x1, x2, 'rx')
    x1, x2 = data2.transpose()
    plt.plot(x1, x2, 'bx')
    
    # サポートベクトルを描画
    data = np.vstack( (data1,data2) )
    plt.scatter(data[svm.support_index,0], data[svm.support_index,1], s=80, c='c', marker='o')

    # 識別境界を描画
    w = hsvm.calc_w(svm)
    x1 = np.linspace(-6, 6, 1000)
    x2 = [hsvm.f(x, w, svm.support_b) for x in x1]
    plt.plot(x1, x2, 'g-')

    #各地点でのyをコンター表示
    contour_division = 50
    seq = np.linspace(-6, 6, contour_division)
    x1list, x2list = np.meshgrid(seq, seq)

    x1x2list = []
    for x1,x2 in zip(x1list,x2list):
        x1x2list.append(zip(x1,x2))

    #classで色分け
    # zlist = [svm.classify(z, 'class') for z in x1x2list]
    # contour_step = 1
    
    #scoreで色分け
    zlist = [svm.classify(z, 'score') for z in x1x2list]
    contour_step = 100

    plt.contourf(x1list, x2list, zlist, contour_step, cmap=plt.cm.spectral)
    plt.colorbar()
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.show()
