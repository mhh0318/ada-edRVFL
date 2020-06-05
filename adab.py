import numpy as np
from MRVFLtrain import *
from MRVFLpredict import *

def error_rate(y,pred):
    return sum(y!= pred)/len(y)

def adaboost(X_train,y_train,X_test,y_test,M,RVFLtrain,RVFLtest,option):
    w = np.ones(len(X_train))/len(X_train)
    #刚开始总的分类器都是0
    n_train = len(X_train)
    n_test = len(y_train)
    pred_train,pred_test = list(np.zeros(n_train)),list(np.zeros(n_test))
    for i in range(M):
        w1 = w*n_train
        [Model,TrainAcc,TrainingTime] = RVFLtrain(X_train,y_train,option)
        [TestAcc, TestingTime] = RVFLtest(testX, testY, Model)
        y_train_i = clf.predict(X_train)
        y_test_i = clf.predict(X_test)

        # miss is 8.1(b) 中的计算分类误差率要乘以w的
        miss = [int(i) for i in (y_train_i != y_train)]

        # miss2 是8.5中y*G(m)
        miss1 = [x if x == 1 else -1 for x in miss]
        #要注意np.dot()也可以一个ndarry 一个列表相乘 这里计算分类误差率和alpha_m
        error_m =np.dot(w,miss)
        #print(error_m)
        alpha_m = 0.5 *np.log((1-error_m)/error_m)
        #更新权重
        w = np.multiply(w,np.exp([-alpha_m * x for x in miss1 ]))

        #ensemble
        pred_train = [sum(x) for x in zip(pred_train,[alpha_m * i for i in y_train_i ])]
        pred_test = [sum(x) for x in zip(pred_test,[alpha_m * i for i in y_test_i ])]
    pred_train,pred_test = np.sign(np.array(pred_train)),np.sign(np.array(pred_test))
    return error_rate(pred_train,y_train), error_rate(pred_test,y_test)

