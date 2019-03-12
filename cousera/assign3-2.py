### Assignment 3-2: Neural Networks ###
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import io
import matplotlib.cm as cm
import scipy.misc

## 2.1 Model representation ##
all_data = io.loadmat("d:/data/ex03/ex3data1.mat")
x_data = all_data['X'] 
y_data = all_data['y'] # (5000, 1)
y_data = np.array(y_data).reshape(len(y_data), 1)

all_data2 = io.loadmat("d:/data/ex03/ex3weights.mat")
W1 = all_data2['Theta1'].T # (401, 25)
W2 = all_data2['Theta2'].T # (10, 26)

## 2.2 Feedforward Propagation and Prediction ##

def sigmoid(z):
    return (1. / (1 + np.exp(-z)))

def hypothesis(theta, X):
    return sigmoid(np.dot(X, theta))

def add_x0(x):
    return np.insert(x, 0, 1, axis=1)


a1 = np.insert(x_data, 0, 1, axis=1) # (5000, 401)
a2 = add_x0(hypothesis(W1, a1))
y_hat = hypothesis(W2, a2) # (5000, 10)
y_hat = np.argmax(y_hat, axis=1).reshape(len(y_hat), 1)
y_data -= 1
equal = np.equal(y_hat, y_data)
acc = len(equal[equal==True]) / len(y_data)
print(acc) # 97.52%