# -*- coding:utf-8 -*-
import math
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

#隠れ一層ニューラルネットワークによる1出力回帰
class Neural:
    def __init__(self, D, M, K):
        #入力、隠れ層のユニット数はどちらもバイアスユニットを含まない
        self.D = D; #入力ユニット数
        self.M = M; #隠れ層ユニット数
        self.K = K; #出力ユニット数

        # w_MD:入力Xから隠れ層までの重み, w_KM:隠れ層から出力yへの重み
        self.w_ji = np.random.random((M,D+1))
        self.w_kj = np.random.random((M+1))
   
    #活性(activation)を計算
    def activation(self, w, x):
        return np.dot(w, x)
    
    #活性化関数(activation function)hを計算
    def h(self, a, func_type="linear"):
        if func_type == "linear":
            return a
        elif func_type == "tanh":
            return np.tanh(a)
        elif func_type == "logistic":
            return 1 / (1 + np.exp(-a))
        else:
            print "invalid func_type :%s" % func_type
            return

    #特定のデータ点xを順伝播させて出力をもらう xはデータ点でphiが具体的な入力になる 要注意
    def forward_prop(self, x):
        a_j = [self.activation(self.w_ji[j,:], x) for j in range(0, M)]
        self.z   = np.array(self.h(a_j, "tanh"))
        self.z   = np.append(self.z,1)
        a_i = self.activation(self.w_kj, self.z)
        self.y   = np.array(self.h(a_i, "logistic"))

    def back_prop(self, t):
        self.delta_k = self.y-t
        self.delta_j = np.array([ (1-self.z[j]**2)*(self.w_kj[j]*self.delta_k) for j in range(0,M) ])  #バイアスパラメタ注意(z,w_KMはM+1)
        return

    def gradEn(self, x):
        self.delta_wMD = np.tensordot(self.delta_j,x,0)
        self.delta_wKM = self.delta_k*self.z

    def update_w(self, eta):
        self.w_kj -= eta*self.delta_wKM
        self.w_ji -= eta*self.delta_wMD
        
    def start_training(self, X, t, eps):
        eta = 1e-1
        count=0
        while True:
            count+=1
            list = range(N)
            np.random.shuffle(list)
            
            E=0
            for n in list:
                #順伝播
                self.forward_prop(X[n])
                E+= 0.5*((self.y-t[n])**2)
                E-= t[n]*np.log(self.y)+(1-t[n])*np.log(1-self.y)
                
                #逆伝播
                self.back_prop(t[n])
                self.gradEn(X[n])
                
                #パラメータの更新
                self.update_w(eta)

                delta_w_norm = np.linalg.norm(self.delta_wKM.flatten())+np.linalg.norm(self.delta_wMD.flatten())
            
            if count % 100 == 0:
                eta *=0.99
                print "%5d count. error squared = %e\r" % (count, E), 
            if delta_w_norm < eps:
                print "delta_w_norm         : %e" % delta_w_norm
                print "total error squared  : %e" % E
                break

    def classification(self, x):
        a_j = [self.activation(self.w_ji[j,:], x) for j in range(0, M)]
        z   = np.array(self.h(a_j, "tanh"))
        z   = np.append(z,1)
        a_i = self.activation(self.w_kj, z)
        y   = np.array(self.h(a_i, "logistic"))
        return y


#ヘビサイド関数と見せかけた何か
def heaviside(x):
    if x > 0:
        return 1
    if x < 0:
        return -1

#座標から入力データをバイアス付きで生成
def generate_data(x):
    return [x**dim for dim in range(0, D)]


if __name__ == "__main__":
    #D:入力の次元 (x^2, x, 1なので三個), M:隠れユニット数の次元, K:出力の次元, N:データ数
    D = 2; M = 2; K = 1; N = 100

    # 訓練データを作成
    cls1 = []
    cls2 = []
    
    mean1 = [-1, 0]
    mean2 = [1, -1]
    mean3 = [0, 0]
    cov = [[1.0,0.8], [0.8,1.0]]
    
    # ex1.ノイズありデータ作成
    # cls1.extend(np.random.multivariate_normal(mean1, cov, N/2))
    # cls2.extend(np.random.multivariate_normal(mean2, cov, N/2-20))
    # cls2.extend(np.random.multivariate_normal(mean3, cov, 20))

    # ex2.sinデータ作成
    yoko=np.linspace(-2, 0, N/2)
    tate=np.sin(yoko*np.pi/2)+yoko
    cls1=np.vstack((yoko,tate)).T
    
    yoko=np.linspace( 0, 2, N/2)
    tate=np.sin(yoko*np.pi/2)-yoko
    cls2=np.vstack((yoko,tate)).T

    # データ行列X_dataを作成
    temp = np.vstack((cls1, cls2))
    temp2 = np.ones((N, 1))
    X_data = np.hstack((temp, temp2))
 

    # ラベルTを作成（1-of-K表現ではないので注意）
    t = []
    for i in range(N/2):
        t.append(1.0)
    for i in range(N/2):
        t.append(0.0)
    T_data = np.array(t)
    
    neural = Neural(D, M, K)
    neural.start_training(X_data,T_data, 1e-5)
   
    # 訓練データを描画
    x1, x2 = np.array(cls1).transpose()
    plt.plot(x1, x2, 'rx')
    
    x1, x2 = np.array(cls2).transpose()
    plt.plot(x1, x2, 'bo')

    #試しにクラス分類してみる
    print "sample classify:(-0.495, 2) =>", neural.classification(np.array([-0.495,2,1]))
    plt.plot(-0.495, 2,"g+")

    #各地点でのyをコンター表示
    contour_division = 200
    seq = np.linspace(-6, 6, contour_division)
    x1list, x2list = np.meshgrid(seq, seq)
    x3list = np.ones((contour_division,contour_division))

    x123list = []
    for x1,x2,x3 in zip(x1list,x2list,x3list):
       x123list.append(zip(x1,x2,x3))

    zlist = []
    for x123 in x123list:
        temp = []
        for z in x123:
            temp.append(neural.classification(np.array(z)))
        zlist.append(temp)

    plt.contourf(x1list, x2list, zlist, 100)
    plt.colorbar()
    plt.xlim(-3, 3)
    plt.ylim(-5, 5)
    plt.show()
