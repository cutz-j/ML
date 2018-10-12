### Assignment 1 ###

import numpy as np
import matplotlib.pyplot as plt

## Q1: eye matrix ##
def eye(num):
    resList = []
    for i in range(num):
        tempList = []
        for j in range(num):
            if i == j:
                tempList.append(1)
            else:
                tempList.append(0)
        resList.append(tempList)
    return np.array(resList)
#print(eye(5))

## Q2: Linear regression with one variable ##
data = open("d:/data/ML/ex01/ex1data1.txt", encoding='utf-8')
dataList = []
for i in data:
    dataList.append(i.strip('\n').split(','))
data = np.array(dataList, dtype=np.float32)
m = len(data)
x = data[:, 0]
y = data[:, 1].reshape(m, 1)
learning_rate = 0.01
iteration = 1500
# X matrix [1, x]
X = np.ones([m, 2])
X[:, 1] = x
n = len(X[0,:])
# first_W
W = np.zeros([n, 1])


## Q2.1: plot
plt.plot(x, y, 'rx', markersize=10)
plt.grid(True)
plt.show()

## Q2.2: Gradient Descent
def hypothesis(x, W):
    return np.dot(x, W)

def cost(x, W, y):
    global m
    return float(1. / (2 * m) * np.dot((hypothesis(x, W) - y).T, (hypothesis(x, W) - y)))


## Q2.2.3: computing the cost J
#print(cost(X, W, y)) # 32.07273422
    
def descent(x, W, y):
    costList = []
    WList = []
    for i in range(iteration):
        w_tmp = W
#        if i % 100 == 0:
#            print(cost)
        costList.append(cost(X, W, y))
        WList.append(W)
        for j in range(len(w_tmp)):
            w_tmp[j] = W[j] - (learning_rate / m) * np.sum((hypothesis(X, W) - y) * X[:, j].reshape(m, 1))
    return W, costList
    
# reshape 제일 중요
W, costList = descent(X, W, y)
plt.plot(range(len(costList)), costList, 'o')
plt.ylim(3, 8)
plt.show()

## Q2.4: Visualize J
plt.plot(x, y, 'rx', markersize=10)
plt.grid(True)
plt.plot(x, hypothesis(X, W), 'b-')
plt.show()

#prediction [1, 3.5] / [1, 7]
predict1 = hypothesis([1, 3.5], W)
predict2 = hypothesis([1, 7], W)
print(predict1, predict2)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    