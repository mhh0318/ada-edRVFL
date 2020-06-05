import numpy as np

def l2_weights(A_merge,b,C,Nsample):

    if A_merge.shape[1]<Nsample:
        beta = np.matmul(np.matmul(np.linalg.inv(np.identity(A_merge.shape[1])/C+np.matmul(A_merge.T,A_merge)),A_merge.T),b)
    else:
        beta = np.matmul(A_merge.T,np.matmul(np.linalg.inv(np.identity(A_merge.shape[0])/C+np.matmul(A_merge,A_merge.T)),b))

    return beta

