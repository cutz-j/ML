### assignment 4-1 ###
import numpy as np
import pandas as pd
import scipy.io
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

def sigmoid(z):
    return 1. / (1 + np.exp(-z))

def layer(X, W):
    X = np.insert(X, 0, 1, axis=1) # bias 추가
    return np.insert(sigmoid(logits(X, W)), 0, 1, axis=1)

def hypothesis(X, W):
    return sigmoid(logits(X, W))  

def cost(X, y, hypothesis, ld=0):
    m = len(X)
    hypo = hypothesis
    cost_sum = 0
    for i in range(m):
        cost_k = 0
        for k in range(classes):
            error = -(y[i][k] * np.log(hypo[i][k])) - ((1 - y[i][k]) * np.log(1 - (hypo[i][k])))
            cost_k += error
        cost_sum += cost_k / m
    return cost_sum

hypo = hypothesis(layer(X, W1), W2)
err = cost(X, y, hypo) # 0.287629165

## 1.4 Regularized cost function ##
def reg_cost(X, y, hypothesis, ld=1):
    m = len(X)
    hypo = hypothesis
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
        
err_reg = reg_cost(X, y, hypo) # 0.384487796

### 2 backpropagation ###
## 2.1 Sigmoid gradient ##
def sigmoidGradient(z):
    ## sigmoid 미분 ##
    return sigmoid(z)*(1 - sigmoid(z))

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
    m = len(X)
    ## frontpropagation ##
    a1 = np.insert(X, 0, 1, axis=1)
    z2 = logits(a1, W1)
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = logits(a2, W2)
    hypothesis = sigmoid(z3)
    
    # delta matrix # --> cost
    Delta2 = np.zeros_like(W2)
    Delta1 = np.zeros_like(W1)
    
    # cost #
    for i in range(m):
        delta3 = hypothesis[[i]] - y[[i]] # 편미분 (partial J / partial a3) * (partial a3 / partial z3)
        Delta2 += np.dot(a2[[i]].T, delta3) # (1,26).T *(1,10) # partial z3 / partial w2 완성
        # 편미분 (partialJ(2) / partial a2) * (partial a2 / partial z2) #
        delta2 = np.matmul(delta3, W2.T[:,1:]) * sigmoidGradient(z2)[i] # (1,10) (10, 26)
        Delta1 += np.dot(a1[[i]].T, delta2) # 

    Delta2 *= 1/m
    Delta1 *= 1/m
    # reg
    Delta1[:,1:] = Delta1[:,1:] + (float(ld)/m) * W1[:,1:]
    Delta2[:,1:] = Delta2[:,1:] + (float(ld)/m) * W2[:,1:]
    return [Delta1, Delta2]
        
back = backpropagation(W_init, X, y, ld=0.) 
      
## 2.4 gradient checking ##
hypo = hypothesis(layer(X, W1), W2)
err = cost(X, y, hypo) # 0.287629165
 
 
## 2.5 Regularized NN ##
## 2.6 run/learn/predict ##
def gradientDescent(W, X, y, ld=0.):
    W1 = W[0]
    W2 = W[1]
    for i in range(100):
        hypo = hypothesis(layer(X, W1), W2)
        cost_val = reg_cost(X, y, hypo, ld=0.001)
        Delta = backpropagation(W, X, y, ld=0.001)
        W1 -= Delta[0]
        W2 -= Delta[1]
        if i % 10 == 0:
            print(cost_val)
    return hypo, [W1, W2]
    
    
y_hat, err = gradientDescent(W_init, X, y, ld=0.001)
predict = np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1))
acc = len(y[predict]) / len(y)
print(acc)