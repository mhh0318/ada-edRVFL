import numpy as np
import time
from function import *
from l2_weights import *
from majorityVoting import *
from model import model as mod


def MRVFLtrain(trainX,trainY,option):
    
    rand_seed= np.random.RandomState(option.seed)

    [n_sample,n_dims] = trainX.shape
    N = option.N
    L = option.L
    C = option.C
    s = option.scale   #scaling factor
    A=[]
    beta=[]
    weights = []
    biases = []
    mu = []
    sigma = []


    A_input= trainX


    time_start=time.time()
    ada_weights = np.ones((L,n_sample))
    classifier_weights = np.ones(L)
    ada_weight = np.expand_dims(np.ones(len(trainX))/len(trainX),axis=1)
    pred_idx = []

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
        # trainX *= ada_weight * n_sample
        A_tmp = np.concatenate([trainX,A_,np.ones((n_sample,1))],axis=1)
        beta_=l2_weights(A_tmp,trainY,C,n_sample)

        A.append(A_tmp)
        beta.append(beta_)

        #clear A_ A_tmp A1_temp2 beta_

        trainY_temp=np.matmul(A_tmp,beta_)
        n_classes = trainY.shape[1]
        prob = np.expand_dims(softmax(trainY_temp), axis=1)
        pred_num = np.argmax(prob,axis=2).ravel()
        pred_oh = np.zeros((pred_num.size,n_classes))
        trainY_num = np.argmax(trainY,axis=1).ravel()
        mask = np.array(pred_num!=trainY_num)[:,np.newaxis]
        err =np.sum(ada_weight*mask) / np.sum(ada_weight)
        classifier_weight = np.log((1-err)/err)+np.log(n_classes-1)
        classifier_weights[i] = classifier_weight
        ada_weight*=np.exp(classifier_weight*mask)
        ada_weight/=ada_weight.sum()
        ada_weights[i] = ada_weight.ravel()
        if err > (n_classes - 1 / n_classes):
            classifier_weight = 0
            ada_weight = np.expand_dims(np.ones(len(trainX))/len(trainX),axis=1)
            print('Error rate of {} Layer is higher than random, drop.'.format(i) )
        pred_oh[range(n_sample),pred_num] = classifier_weight
        pred_idx.append(pred_oh)

        trainX *= ada_weight*n_sample
        A_input = np.concatenate([trainX, A_], axis=1)


    #pred = sum(samm_prob) / L
    pred_idx = np.array(pred_idx)
    pred = np.argmax(pred_idx.sum(axis=0),axis=1)
    time_end = time.time()
    Training_time = time_end-time_start


    ## Calculate the training accuracy

    TrainingAccuracy = np.sum(pred== np.argmax(trainY,axis=1).ravel())/n_sample



    model = mod(L,weights,biases,beta,mu,sigma,classifier_weights)
        
    return model,TrainingAccuracy,Training_time

