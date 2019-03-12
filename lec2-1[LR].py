### Linear Regression: Single / Vectorization ###
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([4.0391, 1.3197, 9.5613, 0.5978, 3.5316, 0.1540, 1.6899, 7.3172, 4.5092, 2.9632]).reshape(10,1)
y_data = np.array([11.4215, 10.0112, 30.2991, 1.0625, 13.1776, -3.1976, 6.7367, 23.8550, 14.8951, 11.6137]).reshape(10,1)

W = np.random.normal(size=(x_data.shape[1], y_data.shape[1])) # shape (1, 1)
b = np.random.normal(size=(1, 1))

X_data = np.insert(x_data, 0, 1., axis=1) # bias 항 추가
W_vec = np.random.normal(size=(X_data.shape[1], y_data.shape[1])) # shape(2, 1) # W = [w0, w1]

## parameter ##
m = X_data.shape[0] # 10 --> data 수
learning_rate = 0.01 
iteration = 1000 # 학습 횟수

def hypothesis(x, w, b):
    ## hypothesis: linear regression ##
    return w * x + b

def hypothesis_vector(X, W):
    ## vectorized hypothesis ##
    return np.matmul(X, W)

def cost(x, w, b, y):
    ## cost: MLE ##
    return 1. / 2 * m * np.sum(np.square(y - hypothesis(x, w, b)))

def cost_vec(X, W, y):
    ## cost: MLE vectorized ##
    return 1. / 2 * np.mean(np.square(y - hypothesis_vector(X, W)))

def cost_vec_reg(X, W, y, ld=0.1):
    ## cost_vecorized + l2 reg ##
    return cost_vec(X, W, y) + ld * np.sum(np.square(W[1:])) # bias reg X

def gd_vector(X, W, y, learning_rate=0.01, ld=0.0, iteration=1000):
    ## gradient Descent Vector ##
    m = X.shape[0]
    cost_list = []
    for i in range(iteration):
        W_new = W.copy()
        if i % 100 == 0:
            print(cost_vec_reg(X, W, y, ld))
        cost_list.append(cost_vec_reg(X, W, y, ld))
        W = W_new - learning_rate * (1. / m * np.sum((hypothesis_vector(X, W_new) - y) * X) - (2 * ld * W_new))
    return W, cost_list

W_hat, cost_list = gd_vector(X_data, W_vec, y_data, 0.0003, 0.0, 1000)























