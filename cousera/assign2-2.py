import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

### 2. Regularized logistic regression ###

## Variable ##
all_data = pd.read_csv("d:/data/ex02/ex2data2.txt", header=None, sep=',')
accept = all_data[all_data[2] == 1]
reject = all_data[all_data[2] == 0]
m = len(all_data)
X = all_data.iloc[:,:2]
X = np.insert(np.array(X), 0, 1, axis=1)
y = np.array(all_data.iloc[:,2]).reshape(m,1)
iteration = 1000
learning_rate = 0.01

## 2.1 Visualizing the data ##
def plotting():
    plt.plot(accept.iloc[:,0], accept.iloc[:,1], 'ro')
    plt.plot(reject.iloc[:,0], reject.iloc[:,1], 'bx')
    plt.legend
    plt.show()
#plotting()
    
## 2.2 feature mapping ##
def mapFeature(x1, x2):
    degrees = 6
    out = np.ones( (x1.shape[0], 1) )
    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1 ** (i-j)
            term2 = x2 ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
            out   = np.hstack(( out, term ))
    return out

map_X = mapFeature(X[:,1], X[:,2]) # shape(118, 28)

## 2.3 Cost function and gradient ##
def sigmoid(z):
    return (1. / (1 + np.exp(-z)))

def hypothesis(theta, X):
    return sigmoid(np.dot(X, theta))

def cost(theta, X, y, mLambda=0.):
    global m
    zero = np.dot((-y).T, np.log(hypothesis(theta, X)))
    one = np.dot((1 - y).T, np.log(1 - hypothesis(theta, X)))
    reg = (mLambda/(2*m)) * np.sum(np.dot(theta[1:].T, theta[1:]))
    return (1 / m) * (np.sum(zero - one)) + reg

def gradientDescent(theta, X, y, mLambda=0): # all-batch 학습
    global m, iteration, learning_rate
    costList = []
    thetaList = []
    theta_tmp = theta.copy()
    for i in range(iteration):
        costList.append(cost(theta, X, y, mLambda))
        thetaList.append(theta)
        theta_tmp[0] = (1 / m) * np.sum(np.dot((hypothesis(theta, X) - y).T, X[:, 0].reshape(m, 1)))
        for j in range(1, len(theta_tmp)): # len(theta_tmp) ==  n; feature 개수
            theta_tmp[j] = (1 / m) * np.sum(np.dot((hypothesis(theta, X) - y).T, X[:, j].reshape(m, 1))) + (mLambda / m) * theta[j]
        theta = theta_tmp.copy()
    return theta, costList

initial_theta = np.zeros((map_X.shape[1], 1), dtype=np.float32)
test = cost(initial_theta, map_X, y)
#print(test) # 0.6931471805599453

## learning parameters ##
ld = 1

def optimizeTheta(theta, X, y, mylambda=0):
    res = optimize.minimize(cost, theta, args=(X, y, mylambda), method='BFGS',
                        options={"maxiter":500, "disp":False})
    return np.array([res.x]), res.fun

#theta, mincost = optimizeTheta(initial_theta, map_X, y, mylambda=ld)
    
theta, costList = gradientDescent(initial_theta, map_X, y, mLambda=ld)

def plotBoundary(mytheta, myX, myy, mylambda=0.):
    theta, mincost = optimizeTheta(mytheta, myX, myy, mylambda)
    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta, myfeaturesij.T)
    zvals = zvals.transpose()
    u, v = np.meshgrid(xvals, yvals)
    mycontour = plt.contour(xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")

plt.figure(figsize=(12,10))
plt.subplot(221)
plotting()
plotBoundary(theta, map_X, y, 0.)

plt.subplot(222)
plotting()
plotBoundary(theta, map_X, y, 1.)

#plt.subplot(223)
#plotting()
#plotBoundary(theta, map_X ,y,10.)










