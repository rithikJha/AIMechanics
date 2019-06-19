import DeepLearningLibrary.trainMeDeep as train
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
   
    return train_X, train_Y, test_X, test_Y

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = load_dataset()
layers_dims = [train_X.shape[0], 10,5,4,1]
activation_tuple = ("relu","relu","tanh","sigmoid")

parameters = train.model(train_X, train_Y, layers_dims,activation_tuple,learning_rate=0.03, num_iterations = 15000, print_cost = True)