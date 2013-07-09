# -*- coding:utf-8 -*-
import math
import numpy as np
import cvxopt as co
import cvxopt.solvers
import matplotlib.pyplot as plt

#描画用関数 wを計算
def calc_w(svm):
    at = (svm.support_a* svm.support_t)
    x  = svm.support_x
    w  = np.dot(at, x)
    return w

#描画用関数 決定境界を計算
def f(x, w, b):
    return - (w[0] / w[1]) * x - (b / w[1])

#2クラス判別SVM ハードマージン
class hardSVM:
    def train(self, x1, x2):
        N = x1.shape[0] + x2.shape[0]
        x = np.vstack((x1, x2))
        t1 = np.ones(x1.shape[0])
        t2 = -np.ones(x2.shape[0])
        t = np.hstack((t1, t2)).T

        a = self.QuadraticProgramming(N, x, t)
        support_index = self.select_seport_vector(a)
        b = self.calc_bias(x, t, a, support_index)

        self.support_a = a[support_index]
        self.support_x = x[support_index]
        self.support_t = t[support_index]
        self.support_index = support_index
        self.support_b = b

    def QuadraticProgramming(self, N, x, t):
        #objective function (1/2.)x^tQ^tx - a^tx
        temp = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                temp[i, j] = t[i] * t[j] * self.kernel(x[i], x[j])

        Q =  co.matrix(temp)
        p =  co.matrix(-np.ones(N))

        #constraint Gx…h
        G = co.matrix(-np.identity(N))
        h = co.matrix( np.zeros(N))

        # constraint Ax=b
        A = co.matrix(t, (1,N))
        b = co.matrix(0.0)

        sol = cvxopt.solvers.qp(Q, p, G, h, A, b)
        a = np.array(sol['x']).reshape(N)
        return a

    def select_seport_vector(self, a):
        return np.where(a>0.0001)

    #バイアスパラメータbを計算
    def calc_bias(self, x, t, a, support_index):
        s_a = a[support_index]
        s_t = t[support_index]
        s_x = x[support_index]

        sum = 0
        for n in range(len(support_index[0])):
            temp = 0
            for m in range(len(support_index[0])):
                temp += s_a[m] * s_t[m] * self.kernel(s_x[n], s_x[m])
            sum += (s_t[n] - temp)
        b = sum / len(support_index[0])
        return b

    #新しいデータ点xを識別する
    def classify(self, x, type='class'):
        atk = 0
        for n in range(len(self.support_index[0])):
            atk += self.support_a[n]*self.support_t[n]*self.kernel(x, self.support_x[n])
        y = atk + self.support_b

        if(type == 'class'):
            val = np.zeros(y.shape)
            val[np.where(y>=0)] =  1
            val[np.where(y<0)]  = -1
        elif(type == 'score'):
            val = y
        else:
            print '### warning! undefined return_type ###'
            val = 0
        return val

    def kernel(self, x1, x2, kernel_type='linear'):
        #線形カーネル
        if(kernel_type == 'linear'):
            k = np.dot( x1, x2.T)
        elif(kernel_type == 'debug'):
            print x1.shape, x2.shape
            k = np.dot( x1, x2.T)
        else:
            print '### warning! undefined kernel_type ###'
            k = 0
        return k


if __name__ == '__main__':
    N = 200

    mu1    = [-1, -2]
    mu2    = [1, 3]
    sigma = [[1.0, 0.8], [0.8, 1.0]]

    data1 = np.random.multivariate_normal(mu1, sigma, N/2)
    data2 = np.random.multivariate_normal(mu2, sigma, N/2)

    svm = hardSVM()
    svm.train(data1, data2)

    #クラス判別
    y = svm.classify( (-1, 2) )
    if(y>=0):
        print(y,1)
    else:
        print(y,-1)    

    # 訓練データを描画
    x1, x2 = data1.transpose()
    plt.plot(x1, x2, 'rx')
    x1, x2 = data2.transpose()
    plt.plot(x1, x2, 'bx')
    
    # サポートベクトルを描画
    data = np.vstack( (data1,data2) )
    plt.scatter(data[svm.support_index,0], data[svm.support_index,1], s=80, c='c', marker='o')
 
    # 識別境界を描画
    w = calc_w(svm)
    x1 = np.linspace(-6, 6, 1000)
    x2 = [f(x, w, svm.support_b) for x in x1]
    plt.plot(x1, x2, 'g-')
    
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.show()

