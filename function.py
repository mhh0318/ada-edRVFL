import numpy as np
import numpy.matlib

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    num=np.exp(x)
    dem=np.sum(num,axis=1).reshape(-1,1)
    dem=numpy.matlib.repmat(dem,1,x.shape[1])
    return num/dem

def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*np.where(x >= 0, x, alpha*(np.exp(x)-1))