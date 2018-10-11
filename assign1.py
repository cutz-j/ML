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
data = open("d:/data/ML/ex01/ex1data1.txt", "r")
dataList = []
for line in data:
    dataList.append(data.readline().strip().split(','))
dataList.remove([''])
data = np.array(dataList, dtype=np.float32)

x = data[:, 0]
y = data[:, 1]
m = len(data)
learning_rate = 0.01

# 2.1: plot
plt.plot(x, y, 'rx', markersize=10)
plt.grid(True)
plt.show()

# 2.2: Gradient Descent

def hypothesis(x, W):
    return np.dot(x, W)

def cost(x, W, y):
    global m
    return 1 / 2 * m * np.dot(hypothesis(x, W).T, hypothesis(x, W))

def descent(x, W, y):
    global m, learning_rate
    return W - learning_rate * cost(x, W, y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    