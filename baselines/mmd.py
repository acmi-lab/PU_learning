#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import torch


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss

# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _get_exponent(Z):
    diag_Z = np.expand_dims(np.diag(Z), 1) 
    exponent = diag_Z - 2 * Z + diag_Z.T
    return exponent

def _mix_rbf_kernel(X, Y, sigma_list):
    XXT = np.matmul(X, X.T)
    YYT = np.matmul(Y, Y.T)
    XYT = np.matmul(X, Y.T)

    XXT_exp = _get_exponent(XXT)
    YYT_exp = _get_exponent(YYT)
    XYT_exp = np.expand_dims(np.diag(XXT),1) - 2*XYT + np.expand_dims(np.diag(YYT),0)

    K_XX = 0.0
    K_YY = 0.0
    K_XY = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K_XX += np.exp(-gamma * XXT_exp)
        K_YY += np.exp(-gamma * YYT_exp)
        K_XY += np.exp(-gamma * XYT_exp)

    return K_XX, K_XY, K_YY, len(sigma_list)


def pre_compute_kernels(X,Y, sigma_list=[1.0]):

    return _mix_rbf_kernel(X,Y,sigma_list)


def mix_rbf_mmd2(K_XX, K_XY, K_YY, probs, biased=True):

    return _mmd2(K_XX, K_XY, K_YY, probs, biased=biased)


def _mmd2(K_XX, K_XY, K_YY, probs, biased=False):
    n = K_XX.shape[0]   
    m = K_YY.shape[0]

    diag_X = torch.diag(K_XX)                       # (m,)
    diag_Y = torch.diag(K_YY)                       # (m,)
    sum_diag_X = torch.sum(diag_X)
    sum_diag_Y = torch.sum(torch.mul(diag_Y, torch.mul(probs,probs)))

    K_XX_sums = K_XX.sum() - sum_diag_X
    K_YY_sums = torch.matmul(torch.unsqueeze(probs,axis=0), torch.matmul(K_YY, torch.unsqueeze(probs,axis=1))).sum() - sum_diag_Y
    K_XY_sums = torch.matmul(K_XY, torch.unsqueeze(probs,axis=1)).sum()


    # print((K_XX_sums + sum_diag_X) / (n * n))
    # print((K_YY_sums + sum_diag_Y) / (m * m))
    # print(2.0 * K_XY_sums / (m * n))
    if biased:
        denom_YY_sums = torch.matmul(torch.unsqueeze(probs,axis=1), torch.unsqueeze(probs,axis=0)).sum()
        denom_XY_sums = probs.sum()
        mmd2 = ((K_XX_sums + sum_diag_X) / (n * n)
            + (K_YY_sums + sum_diag_Y) / denom_YY_sums
            - 2.0 * K_XY_sums / ( denom_XY_sums * n))
    else:
        mmd2 = (K_XX_sums / (m * (m - 1))
            + K_YY_sums / (m * (m - 1))
            - 2.0 * K_XY_sums / (m * m))

    return mmd2

def check_kernel(X,Y, sigma = 1.0): 
    n = X.shape[0]
    m = Y.shape[0]

    K = np.zeros((n,m))

    for i in range(n): 
        for j in range(m): 
            gamma = 1.0 / (2 * sigma**2)
            K[i][j] = np.exp(-gamma * (np.linalg.norm(X[i] - Y[j])**2) )
    
    return K

