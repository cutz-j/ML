### assignment 4-1 ###
import numpy as np
import pandas as pd
import scipy

## 1.1 visual ##
all_data = scipy.io.loadmat("d:/data/ex04/ex4data1.mat")

X = all_data['X'] # shape (5000, 400)
y = all_data['y'] # shape (5000, 1)

w_set = scipy.io.loadmat("d:/data/ex04/ex4weights.mat")
W1 = w_set['Theta1'].T # shape (401, 25)
W2 = w_set['Theta2'].T # shape (26, 10)

## 1.2 model representation ##
## 1.3 feedforward and cost function

# parameter #
classes = 10

def one_hot(y):
    one_hot = np.zeros((len(y), classes), dtype=np.int32)
    for i in range(len(y)):
        one_hot[i, y[i]-1] = 1
    return one_hot

y = one_hot(y)

def logits(X, W):
    return np.matmul(X, W)

def sigmoid(X, W):
    return 1. / (1 + np.exp(-logits(X, W)))

def layer(X, W):
    X = np.insert(X, 0, 1, axis=1) # bias 추가
    return np.insert(sigmoid(X, W), 0, 1, axis=1)

def hypothesis(X, W):
    return sigmoid(X, W)  

def cost(X, y, ld=0):
    m = len(X)
    hypo = hypothesis(layer(X, W1), W2)
    cost_sum = 0
    for i in range(m):
        cost_k = 0
        for k in range(classes):
            error = -(y[i][k] * np.log(hypo[i][k])) - ((1 - y[i][k]) * np.log(1 - (hypo[i][k])))
            cost_k += error
        cost_sum += cost_k / m
    return cost_sum

err = cost(X, y) # 0.287629165

## 1.4 Regularized cost function ##
def reg_cost(X, y, ld=1):
    m = len(X)
    hypo = hypothesis(layer(X, W1), W2)
    cost_sum = 0
    for i in range(m):
        cost_k = 0
        for k in range(classes):
            error = -(y[i][k] * np.log(hypo[i][k])) - ((1 - y[i][k]) * np.log(1 - (hypo[i][k])))
            cost_k += error
        cost_sum += cost_k / m
    total_reg = 0.
    total_reg += np.sum(np.square(W1)) + np.sum(np.square(W2))
        #element-wise multiplication
    total_reg *= float(ld)/(2*m)
    total_reg += cost_sum
    return total_reg    
        
err_reg = reg_cost(X, y) # 0.384487796

### 2 backpropagation ###
## 2.1 Sigmoid gradient ##
def sigmoidGradient(X, W):
    ## sigmoid 미분 ##
    return sigmoid(X, W)*(1 - sigmoid(X, W))

#test = sigmoidGradient([0], [1]) # 0.25

## 2.2 Random init ##
def random_init():
    ## W init (-> random uniform)
    epsilon_init = 0.12
    W_init = [np.random.uniform(low=-epsilon_init, high=epsilon_init, size=W1.shape),
              np.random.uniform(low=-epsilon_init, high=epsilon_init, size=W2.shape)]
    return W_init

W_init = random_init()

## 2.3 Backpropagation ##
def backpropagation(W, X, y, ld=0):
    W1 = W[0]
    W2 = W[1]
    
    # delta matrix 
    delta1 = np.zeros_like(W1)
    delta2 = np.zeros_like(W2)
    

    return        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        