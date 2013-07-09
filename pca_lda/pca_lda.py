# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pca
import lda
from matplotlib.mlab import PCA

def projection(x, w):
    z = np.dot(x, w)
    return z

if __name__ == '__main__':
    n1     = 100
    mu1    = [5, 4]
    sigma1 = [[1.0, 0.9], [0.9, 1.0]]

    n2     = 100
    mu2    = [4, 5]
    sigma2 = [[1.0, 0.9], [0.9, 1.0]]

    N1 = np.random.multivariate_normal(mu1, sigma1, n1)
    N2 = np.random.multivariate_normal(mu2, sigma2, n2)

    DATA = np.vstack( (N1, N2) )
    data_mu = np.mean(DATA, axis=0)

    #PCA
    w_p = pca.PCA(DATA) # original
    #w_p = PCA(DATA).Wt  # mlab liblary
    w_p0 = w_p[0,:] #eig_vec of max eig_val

    pca_line_offset = data_mu[1]- data_mu[0]*w_p0[1]/w_p0[0]
    z = projection(DATA, w_p0)
    print "PCA var:",np.var(z)

    #LDA
    w_l = lda.LDA(N1,N2)
    w_l0 = w_l[0,:] #eig_vec of max_eig_val
    lda_line_offset = data_mu[1]- data_mu[0]*w_l0[1]/w_l0[0]
    z1 = projection(N1, w_l0)
    z2 = projection(N2, w_l0)
    print "LDA var1:",np.var(z1),",var2:",np.var(z2)

    #plot
    pca_x_list = np.arange(2, 8, 0.1)
    lda_x_list = np.arange(3, 6, 0.1)

    pc_0 = [ w_p0[1]/w_p0[0]*x + pca_line_offset for x in pca_x_list]
    lda_0 = [ w_l0[1]/w_l0[0]*x + lda_line_offset for x in lda_x_list]

    lda_0 = [ w_l0[1]/w_l0[0]*x + lda_line_offset for x in lda_x_list]
    plt.scatter(N1[:, 0],N1[:, 1], c='b', marker='x')
    plt.scatter(N2[:, 0],N2[:, 1], c='r', marker='o')
    plt.plot(pca_x_list, pc_0, 'b-')
    plt.plot(lda_x_list, lda_0, 'r-')

    #plot config
    plt.title('The Result of PCA and LDA')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(1,8)
    plt.ylim(1,8)
    plt.legend(('PCA direction', 'LDA direction'))
    plt.show()
