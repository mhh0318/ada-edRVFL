from MRVFLtrain import *
from MRVFLpredict import *

def MRVFL(trainX,trainY,testX,testY,option):

    # Train RVFL
    [Model,TrainAcc,TrainingTime] = MRVFLtrain(trainX,trainY,option)


    # Using trained model, predict the testing data
    [TestAcc,TestingTime] = MRVFLpredict(testX,testY,Model)

    return Model,TrainAcc,TestAcc,TrainingTime,TestingTime
#EOF

