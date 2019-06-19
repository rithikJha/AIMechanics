import DeepLearningLibrary.activations as act
import numpy as np

def intialize_parameters(layers_dims):
    np.random.seed(3)
    L =len(layers_dims)
    parameters={}
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))

    return parameters


def linear_forward(A_prev,W,b):
    Z=np.dot(W,A_prev)+b
    linear_cache =(A_prev,W,b)
    return Z,linear_cache

def linear_activation_forward(A_prev,W,b,activation):

    if activation=="sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = act.sigmoid(Z)

    if activation=="relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = act.relu(Z)

    if activation=="tanh":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = act.tanh(Z)

    cache = (linear_cache,activation_cache)
    return A,cache

def linear_deep_forward(X,parameters,activation_tuple):

    L=len(parameters)//2
    caches=[]
    A_prev=X

    for l in range(1,L+1):
            A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation_tuple[l-1])
            caches.append(cache)
            A_prev=A

    AL = A_prev
    return AL,caches,activation_tuple


def compute_cost(AL,Y):
    m=Y.shape[1]
    cost = (-1/m)*(np.dot(Y,np.log(AL).T)+np.dot(1-Y,np.log(1-AL).T))
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ,linear_cache):
    A_prev,W,b=linear_cache
    m=A_prev.shape[1]
    dA_prev = np.dot(W.T,dZ)
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ, axis=1,keepdims=True)/m
    return dA_prev,dW,db

def linear_activation_backward(dA,caches,activation):
    linear_cache,activation_cache = caches
    if activation=="sigmoid":
        dZ=act.sigmoid_backwards(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache) 

    elif activation=="relu":
        dZ=act.relu_backwards(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache) 

    elif activation=="tanh":
        dZ=act.tanh_backwards(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache) 

    return dA_prev,dW,db

def linear_deep_backward(AL,Y,caches,activation_tuple):
    L=len(caches)
    grads={}
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)

    dAL = -np.divide(Y,AL)+np.divide(1-Y,1-AL)
    dA=dAL
    for l in reversed(range(L)):
        current_cache = caches[l]
        grads["dA"+str(l)],grads["dW"+str(l+1)],grads["db"+str(l+1)]=linear_activation_backward(dA,current_cache,activation_tuple[l])
        dA=grads["dA"+str(l)]

    return grads

def update_parameters(parameters,grads,learning_rate):
    L=len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters
