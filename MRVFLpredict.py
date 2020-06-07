import numpy as np
import time
import numpy.matlib
from function import *
from majorityVoting import *

def MRVFLpredict(testX,testY,model):


    [Nsample,Nfea]= testX.shape


    w=model.w
    b=model.b
    beta=model.beta
    mu=model.mu
    sigma=model.sigma
    L= model.L
    clf_weights = model.ada_weights

    A=[]
    A_input=testX

    time_start=time.time()

    for i in range(L):
        A1=np.matmul(A_input,w[i])
        A1 = (A1-mu[i])/sigma[i]
        A1 = A1 + np.repeat(b[i], Nsample, 0)
        #A1=relu(A1)
        A1=selu(A1)
        A1_temp1 = np.concatenate([testX,A1,np.ones((Nsample,1))],axis=1)


        A.append(A1_temp1)

        #clear A1 A1_temp1 A1_temp2 beta1
        A_input = np.concatenate([testX,A1],axis=1)
    n_classes = testY.shape[1]
    testY_num = np.argmax(testY, axis=1).ravel()
    pred_idx=[]
    for i in range(L):
        A_temp=A[i]
        beta_temp=beta[i]
        clf_weight = clf_weights[i]
        testY_temp=np.matmul(A_temp,beta_temp)
        prob = np.expand_dims(softmax(testY_temp), axis=1)
        pred_num = np.argmax(prob,axis=2).ravel()
        pred_oh = np.zeros((pred_num.size,n_classes))
        pred_oh[range(Nsample), pred_num] = clf_weight
        pred_idx.append(pred_oh)


    pred = np.zeros((L,Nsample,n_classes))
    pred_idx = np.array(pred_idx)
    for i in range(1,L+1):
        pred[i-1] = pred_idx[:i].sum(axis=0)
    predictions = np.argmax(pred,axis=2)

    TestingAccuracy = np.sum(predictions == testY_num, axis=1) / Nsample

    time_end=time.time()

    Testing_time = time_end-time_start

    return TestingAccuracy,Testing_time


