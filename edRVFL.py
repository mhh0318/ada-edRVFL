# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/8 1:36
@author: merci
"""
import os
import numpy as np
import numpy as cp
from option import option as op
from MRVFL import *
import time

root_path = '/home/hu/eRVFL/UCIdata'
data_name = 'statlog-german-credit'
# n_device = 7
print('Dataset Name:{}\nDevice Number:#CPU'.format(data_name))

# cp.cuda.Device(n_device).use()
# load dataset
# dataX
datax = np.loadtxt('{0}/{1}/{1}_py.dat'.format(root_path, data_name), delimiter=',')
dataX = cp.asarray(datax)
# dataY
datay = np.loadtxt('{}/{}/labels_py.dat'.format(root_path, data_name), delimiter=',')
dataY = cp.asarray(datay)

# Validation Index
Validation = np.loadtxt('{}/{}/validation_folds_py.dat'.format(root_path, data_name), delimiter=',')
validation = cp.asarray(Validation)

# Folds Index
Folds_index = np.loadtxt('{}/{}/folds_py.dat'.format(root_path, data_name), delimiter=',')
folds_index = cp.asarray(Folds_index)

types = cp.unique(dataY)
n_types = types.size
n_CV = folds_index.shape[1]
# One hot coding for the target
dataY_tmp = cp.zeros((dataY.size, n_types))
for i in range(n_types):
    for j in range(dataY.size):  # remove this loop
        if dataY[j] == types[i]:
            dataY_tmp[j, i] = 1

option = op(N=256, L=32, C=2 ** -6, scale=1, seed=1, nCV=0)
N_range = [256, 512, 1024]
# N_range = [64, 128, 256, 512]
# N_range = [16, 32, 64]
L = 32
option.scale = 1
# C_range = np.append(0,2.**np.arange(-6, 12, 2))
C_range = 2.**np.arange(-6, 12, 2)

Models_tmp = []
Models = []
# dataX = rescale(dataX) #####delete

train_acc_result = cp.zeros((n_CV, 1))
test_acc_result = cp.zeros((n_CV, 1))
train_time_result = cp.zeros((n_CV, 1))
test_time_result = cp.zeros((n_CV, 1))

MAX_acc = 0
option_best = op(N=256, L=32, C=2 ** -6, scale=1, seed=0)
for i in range(n_CV):
    MAX_acc = 0
    train_idx = cp.where(folds_index[:, i] == 0)[0]
    test_idx = cp.where(folds_index[:, i] == 1)[0]
    trainX = dataX[train_idx, :]
    trainY = dataY_tmp[train_idx, :]
    testX = dataX[test_idx, :]
    testY = dataY_tmp[test_idx, :]
    st = time.time()
    for n in N_range:
        option.N = n
        for j in C_range:
            option.C = j
            sto = time.time()
            option.L = 32
            train_idx_val = cp.where(validation[:, i] == 0)[0]
            test_idx_val = cp.where(validation[:, i] == 1)[0]
            trainX_val = dataX[train_idx_val, :]
            trainY_val = dataY_tmp[train_idx_val, :]
            testX_val = dataX[test_idx_val, :]
            testY_val = dataY_tmp[test_idx_val, :]
            [model_tmp, train_acc_temp, test_acc_temp, training_time_temp, testing_time_temp] = MRVFL(trainX_val, trainY_val, testX_val, testY_val, option)
            if test_acc_temp.max() > MAX_acc:
                MAX_acc = test_acc_temp.max()
                option_best.acc_test = test_acc_temp.max()
                option_best.acc_train = train_acc_temp.max()
                option_best.C = option.C
                option_best.N = option.N
                option_best.L = option.L
                option_best.nCV = i
                print('Temp Best Option:{}'.format(option_best.__dict__))
            print('Training Time for one option set:{:.2f}'.format(time.time()-sto))
            if time.time()-sto>10:
                print('current settings:{}'.format(option.__dict__))
    [model_RVFL, train_acc0, test_acc0, train_time0, test_time0] = MRVFL(trainX, trainY, testX, testY, option_best)
    print('Training Time for one fold set:{:.2f}'.format(time.time() - st))
    Models.append(model_RVFL)
    train_acc_result[i] =train_acc0.max()
    test_acc_result[i] =test_acc0.max()
    train_time_result[i] =train_time0
    test_time_result[i] =test_time0
    del model_RVFL
    print('Best Train accuracy in fold{}:{}\nBest Test accuracy in fold{}:{}'.format(i, train_acc_result[i], i,
                                                                                     test_acc_result[i]))

mean_train_acc = train_acc_result.mean()
mean_test_acc = test_acc_result.mean()
print('Train accuracy:{}\nTest accuracy:{}'.format(train_acc_result, test_acc_result))
print('Mean train accuracy:{}\nMean test accuracy:{}'.format(mean_train_acc, mean_test_acc))
