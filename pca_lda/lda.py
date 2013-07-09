import numpy as np

#linear discriminant analysis(2 class)
# input : [[data1a,data1b,data1c,...],[data2a,data2b,data2c,...],...]
# return: 
def LDA(x, y):
    x_bar = np.mean(x, axis=0)
    y_bar = np.mean(y, axis=0)
    d_bar = (x_bar+y_bar)/2.0

    S1 = np.dot( (x-x_bar).T, (x-x_bar))
    S2 = np.dot( (y-y_bar).T, (y-y_bar))

    Sw = S1+S2
    Sw_inv = np.linalg.inv(Sw)

    Sb = len(x)*np.outer((x_bar-d_bar), (x_bar-d_bar).T) + len(y)*np.outer((y_bar-d_bar), (y_bar-d_bar).T)

    Sw_inv_Sb = np.dot(Sw_inv, Sb)

    # one eig_vec is in one column 
    (eig_val, eig_vec) = np.linalg.eig(Sw_inv_Sb)
    eig_vec = eig_vec.T

    idx = eig_val.argsort()
    idx = idx[::-1]
    
    return eig_vec[idx]