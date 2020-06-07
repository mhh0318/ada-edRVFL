import numpy as cp

def l2_weights(X, target, ridge_parameter, n_sample):

    if ridge_parameter == 0:
        beta = cp.matmul(cp.linalg.pinv(X),target)
    elif X.shape[1]<n_sample:
        beta = cp.matmul(cp.matmul(cp.linalg.inv(cp.eye(X.shape[1]) / ridge_parameter + cp.matmul(X.T, X)), X.T), target)
    else:
        beta = cp.matmul(X.T, cp.matmul(cp.linalg.inv(cp.eye(X.shape[0]) / ridge_parameter + cp.matmul(X, X.T)), target))

    return beta
