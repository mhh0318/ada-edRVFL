import numpy as np
from scipy import stats
 

def majorityVoting(Y,pred_idx):

    Nsample= Y.shape[0]
    Ind_corrClass=np.argmax(Y,axis=1)
    indx=np.zeros(Nsample)
    for i in range (Nsample):
        Y=pred_idx[i,:]
        indx[i]=stats.mode(Y)[0][0] 
        
    acc=np.mean(indx==Ind_corrClass)

    return acc
        