import numpy as np
import time
import numpy.matlib
from function import *
from l2_weights import *
from majorityVoting import *
from model import model as mod
from scipy.special import xlogy


def MRVFLtrain(trainX,trainY,option):
    
    rand_seed= np.random.RandomState(option.seed)

    [n_sample,n_dims] = trainX.shape
    N = option.N
    L = option.L
    C = option.C
    s = option.scale   #scaling factor
    pred_idx=np.ones((L,n_sample))
    A=[]
    beta=[]
    weights = []
    biases = []
    mu = []
    sigma = []
    samm_prob = []

    A_input= trainX


    time_start=time.time()
    ada_weights = np.ones((L,n_sample))
    ada_weight = np.expand_dims(np.ones(len(trainX))/len(trainX),axis=1)

    for i in range(L):

        if i==0:
            w = s*2*rand_seed.rand(n_dims,N)-1

        else:
            w = s*2*rand_seed.rand(n_dims+N,N)-1

        b = s*rand_seed.rand(1,N)
        weights.append(w)
        biases.append(b)

        A_ = np.matmul(A_input,w)
        # layer normalization
        A1_mean = np.mean(A_,axis=0)
        A1_std = np.std(A_,axis=0)
        A_ = (A_-A1_mean)/A1_std
        mu.append(A1_mean)
        sigma.append(A1_std)

        A_ = A_ + np.repeat(b, n_sample, 0)
        # A_ = relu(A_)
        A_ = selu(A_)
        A_tmp = np.concatenate([trainX,A_,np.ones((n_sample,1))],axis=1)
        beta_=l2_weights(A_tmp,trainY,C,n_sample)

        A.append(A_tmp)
        beta.append(beta_)

        #clear A_ A_tmp A1_temp2 beta_

        trainY_temp=np.matmul(A_tmp,beta_)
        n_classes = np.unique(trainY).size
        prob = np.expand_dims(softmax(trainY_temp), axis=1)
        h = (n_classes - 1) * (np.log(prob) - (1. / n_classes) * np.log(prob).sum(axis=1)[:, np.newaxis])
        #indx=np.argmax(trainY_temp,axis=1)
        #indx=indx.reshape(n_sample,1)
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(np.unique(trainY) == trainY[:, np.newaxis])
        estimator_weight = (-1.
                            * ((n_classes - 1.) / n_classes)
                            * xlogy(y_coding, prob).sum(axis=2))
        ada_weight *= np.exp(estimator_weight *((ada_weight > 0) |(estimator_weight < 0)))
        ada_weight /= ada_weight.sum()
        samm_prob.append(h)
        ada_weights[i] = ada_weight.ravel()

        trainX *= ada_weight*n_sample
        A_input = np.concatenate([trainX, A_], axis=1)


    pred = sum(samm_prob) / L

    time_end = time.time()
    Training_time = time_end-time_start


    ## Calculate the training accuracy

    TrainingAccuracy = np.sum(np.argmax(pred,axis=2).ravel() == np.argmax(trainY,axis=1).ravel())/n_sample



    model = mod(L,weights,biases,beta,mu,sigma,ada_weights)
        
    return model,TrainingAccuracy,Training_time

