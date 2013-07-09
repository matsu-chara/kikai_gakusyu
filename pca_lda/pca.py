import numpy as np

#principal component analysis
# input : [[data1a,data1b,data1c,...],[data2a,data2b,data2c,...],...]
# return: [[1st principal component eig_vector],[ 2nd pc.eig_vector],[3rd pc.eig_vector]...]
def PCA(x):
    x_bar = x.mean(axis=0) #mean of x
    m = x-x_bar
    # C = np.cov( m.T, bias=1) 
    # C = np.dot((x - x_bar).T, (x - x_bar))/(len(x)-1) #covarian ce matrix
    C = np.dot(m.T, m) #covariance matrix

    # one eig_vec is in one column 
    (eig_val, eig_vec) = np.linalg.eig(C)
    eig_vec = eig_vec.T
    
    idx = eig_val.argsort()
    idx = idx[::-1]
    return eig_vec[idx]
