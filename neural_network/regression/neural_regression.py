# -*- coding:utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt

#隠れ一層ニューラルネットワークによる1出力回帰
class Neural:
    def __init__(self, D, M, K):
        #入力、隠れ層のユニット数はどちらもバイアスユニットを含まない
        self.D = D; #入力ユニット数
        self.M = M; #隠れ層ユニット数
        self.K = K; #出力ユニット数

        # w_MD:入力Xから隠れ層までの重み, w_KM:隠れ層から出力yへの重み
        self.w_ji = np.random.standard_normal((M,D+1))
        self.w_kj = np.random.standard_normal(M+1)
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
        else:
            print "invalid func_type :%s" % func_type
            return

    #特定のデータ点xを順伝播させて出力をもらう xはデータ点でphiが具体的な入力になる 要注意
    def forward_prop(self, x):
        a_j = [self.activation(self.w_ji[j,:], x) for j in range(0, M)]
        self.z   = np.array(self.h(a_j, "tanh"))
        self.z   = np.append(self.z,1)
        a_i = self.activation(self.w_kj, self.z)
        self.y   = np.array(self.h(a_i, "linear"))

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
                
                #逆伝播
                self.back_prop(t[n])
                self.gradEn(X[n])
                
                #パラメータの更新
                self.update_w(eta)

                delta_w_norm = np.linalg.norm(self.delta_wKM.flatten())+np.linalg.norm(self.delta_wMD.flatten())
            
            if count % 100 == 0:
                eta *=0.999
                print "%d count. error squared = %e\r" % (count, E), 
            if delta_w_norm < eps:
                print "delta_w_norm         : %e" % delta_w_norm
                print "total error squared  : %e" % E
                print self.w_ji
                print self.w_kj
                break

    def regression(self, x):
        a_j = [self.activation(self.w_ji[j,:], x) for j in range(0, M)]
        z   = np.array(self.h(a_j, "tanh"))
        z   = np.append(z,1)
        a_i = self.activation(self.w_kj, z)
        y   = np.array(self.h(a_i, "linear"))
        return y
    
#ヘビサイド関数と見せかけた何か
def heaviside(x):
    if x > 0:
        return 1
    if x < 0:
        return -1

def generate_data(x):
    return [x**dim for dim in range(0, D+1)]

if __name__ == "__main__":
    #D:入力の次元 (x^2, x, 1なので三個), M:隠れユニット数の次元, K:出力の次元, N:データ数
    D = 1; M = 3; K = 1; N = 50

    # x:データ点, t:観測値
    start = -1.0; end = -start;
    X = np.arange(start, end+0.001, (end-start)/N)
    #T = np.array( [ 2*x**2-1 for x in X ] )
    #T = np.array( [ np.sin(x*np.pi) for x in X ] )
    T = np.array( [ np.sin(x*np.pi)*x for x in X ] )
    #T = np.array( [ abs(2*x)-1 for x in X ] )
    #T = np.array( [ heaviside(x) for x in X ] )
    
    #入力ベクトルの生成 バイアスも入れる
    X_data = np.array([generate_data(x) for x in X])
    T_data = T
    
    neural = Neural(D, M, K)
    neural.start_training(X_data,T_data, 1e-4)
    
    #出力
    X_list      = np.arange(start, end, 0.001)
    X_list_data = np.array([generate_data(x) for x in X_list])
    Y_list      = [neural.regression(x) for x in X_list_data]
    plt.plot(X_list, Y_list)
    plt.plot(X, T, 'o')
    plt.xlim(start-0.1,end+0.1)
    plt.ylim(-1.2,1.2)
    
    s=[]
    for x in X_list_data:
        a = np.array([np.dot(neural.w_ji[j], x) for j in range(0, M)])
        z = np.array(np.tanh(a))
        s.append([z[m] for m in range(0,M)])
    
    s = np.array(s)
    for m in range(0,M):
        plt.plot(X_list, s[:,m], "r--")
    
    plt.show()
