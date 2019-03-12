### Assignment 2-1 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

## 변수 선언 ##
all_data = pd.read_csv("d:/data/ex02/ex2data1.txt", sep=',', header=None)
X = all_data.iloc[:,:2]
y = np.array(all_data[2])                # (100, 1)
m = y.size # 100
y = y.reshape(m, 1)
X = np.insert(np.array(X), 0, 1, axis=1) # (100, 3)
initial_theta = np.random.normal(size=(X.shape[1],1)) # (3,1)
iteration = 3000
learning_rate = 0.001615

## 1.1 Visualizing the data ##
admit = all_data.iloc[:,:2][all_data[2]==1]
notadmit = all_data.iloc[:,:2][all_data[2]==0]

def plotting():
    plt.figure(figsize=(10,6))
    plt.plot(admit.iloc[:,0], admit.iloc[:,1], 'ro')
    plt.plot(notadmit.iloc[:,0], notadmit.iloc[:,1], 'bx')
    plt.legend()
    plt.show()
    plt.grid(True)
## 1.2 Implementation ##
## 1.2.1 sigmoid ##

def sigmoid(z):
    return (1. / (1 + np.exp(-z)))

def hypothesis(theta, X):
    return sigmoid(np.dot(X, theta))

def cost(theta, X, y, mLambda=0.):
    global m
    zero = np.dot((y).T, np.log(hypothesis(theta, X)))
    one = np.dot((1 - y).T, np.log(1 - hypothesis(theta, X)))
    reg = (mLambda/(2*m)) * np.sum(np.dot(theta[1:].T, theta[1:]))
    return -(1 / m) * (np.sum(zero + one)) + reg

def gradientDescent(theta, X, y, mLambda=0): # all-batch 학습
    global m, iteration, learning_rate
    costList = []
    thetaList = []
    theta_tmp = theta.copy()
    for i in range(iteration):
        costList.append(cost(theta, X, y, mLambda))
        thetaList.append(theta)
        if i % 100 == 0:
            print(cost(theta, X, y, mLambda))
        theta_tmp[0] = theta[0] - ((learning_rate / m) * np.sum(np.dot((hypothesis(theta, X) - y).T, X[:, 0].reshape(m, 1))))
        for j in range(1, len(theta_tmp)): # len(theta_tmp) ==  n; feature 개수
            theta_tmp[j] = theta[j] - ((learning_rate / m) * np.sum(np.dot((hypothesis(theta, X) - y).T, X[:, j].reshape(m, 1))) + (mLambda / m) * theta[j])
        theta = theta_tmp.copy()
#        print(theta)
    return theta, costList

#theta, mincost = gradientDescent(initial_theta,X,y)
#print(theta, mincost)

initial_theta = np.zeros((X.shape[1], 1), dtype=np.float32)
theta, costList = gradientDescent(initial_theta, X, y, mLambda=0)
print(theta, costList[-1])

## 1.2.3 Decision boundary ##
boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
plotting()
plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
plt.legend()

## 1.2.4 Evaluating logistic regression ##
test = np.array([1., 45, 85])
answer = hypothesis(theta, test)
#print(answer) #0.7762915904112411 --> admit 확률이 0.776

## predict ##
def prediction(theta, X):
    return hypothesis(theta, X) >= 0.5

theta = theta.reshape(3,1)
admit = np.insert(np.array(admit), 0, 1, axis=1)
notadmit = np.insert(np.array(notadmit), 0, 1, axis=1)
p_correct = float(np.sum(prediction(theta, admit)))
n_correct = float(np.sum(np.invert(prediction(theta, notadmit))))
tot = len(admit) + len(notadmit)
correct = float((p_correct + n_correct) / tot)
print(correct) # 0.89 // 총 100개의 데이터 중 89개 정답 *(학습 데이터에서)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


