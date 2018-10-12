### Assignment 1 ###

import numpy as np
import matplotlib.pyplot as plt

## Q3: Linear regression with multiple variable ##
data = open("d:/data/ML/ex01/ex1data2.txt", encoding='utf-8')
dataList = []
for i in data:
    dataList.append(i.strip('\n').split(','))
data = np.array(dataList, dtype=np.float32)
m = len(data)
memoryList = []

# scaling
def standard(x):
    global memoryList
    memoryList.append((np.mean(x), np.std(x)))
    return (x - np.mean(x)) / np.std(x)

def rollback(x, num):
    mean, std = memoryList[num][0], memoryList[num][1]
    return x * std + mean

x1 = data[:, 0]
x1_scale = standard(x1)
x2 = data[:, 1]
x2_scale = standard(x2)
y = data[:, -1]
y = standard(y).reshape(m, 1)
learning_rate = 0.01
iteration = 1500
# X matrix [1, x]
X = np.ones([m, 3])
X[:, 1] = x1_scale
X[:, 2] = x2_scale
n = len(X[0,:])
# first_W
W = np.zeros([n, 1])



## Gradient Descent
def hypothesis(x, W):
    return np.dot(x, W)

def cost(x, W, y):
    global m
    return float(1. / (2 * m) * np.dot((hypothesis(x, W) - y).T, (hypothesis(x, W) - y)))
    
def descent(x, W, y):
    costList = []
    WList = []
    for i in range(iteration):
        w_tmp = W
        if i % 100 == 0:
            print(cost(X, W, y))
        costList.append(cost(X, W, y))
        WList.append(W)
        for j in range(len(w_tmp)):
            w_tmp[j] = W[j] - (learning_rate / m) * np.sum((hypothesis(X, W) - y) * X[:, j].reshape(m, 1))
    return W, costList
    
# reshape 제일 중요
W, costList = descent(X, W, y)
plt.plot(range(len(costList)), costList, 'o')
plt.show()

#X[:, 1] = rollback(X[:, 1], 0)
#X[:, 2] = rollback(X[:, 1], 1)
#y = rollback(y, 2)

#plt.plot(X[:,1:], y, 'rx', markersize=10)
#plt.grid(True)
#plt.plot(X[:,1:], hypothesis(X, W))
#plt.show()
#    

# predict
predict = [1, (1650.-memoryList[0][0])/memoryList[0][1], (3-memoryList[1][0])/memoryList[1][1]]

print(rollback(hypothesis(predict, W), 2))


# Q3.3
def normEqtn(X,y):
    #restheta = np.zeros((X.shape[1],1))
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

data = open("d:/data/ML/ex01/ex1data2.txt", encoding='utf-8')
dataList = []
for i in data:
    dataList.append(i.strip('\n').split(','))
data = np.array(dataList, dtype=np.float32)
m = len(data)
memoryList = []
x1 = data[:, 0]
x2 = data[:, 1]
y = data[:, -1]
X = np.ones([m, 3])
X[:, 1] = x1
X[:, 2] = x2
n = len(X[0,:])
W = np.zeros([n, 1])

print ("$%0.2f" % float(hypothesis(normEqtn(X,y),[1,1650.,3])))    
    
    
    
    
    
    
    
    
    
    
    
    