import numpy as np 

def sigmoid(Z):
    activation_cache = Z
    A= 1/(1+np.exp(-Z))
    return A,activation_cache

def relu(Z):
    activation_cache = Z
    A = np.maximum(0,Z)
    return A,activation_cache

def tanh(Z):
    activation_cache=Z
    A=np.tanh(Z)
    return A,activation_cache

def sigmoid_backwards(dA,activation_cache):
    Z=activation_cache
    A = 1/(1+np.exp(-Z))
    dZ = dA*A*(1-A)
    return dZ

def relu_backwards(dA,activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def tanh_backwards(dA,activation_cache):
    Z=activation_cache
    A=np.tanh(Z)
    dZ=dA*(1-A**2)
    return dZ