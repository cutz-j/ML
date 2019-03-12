### Assignment 6-1. SVM ###
import numpy as np
import pandas as pd
import scipy
import scipy.io
import os
import matplotlib.pyplot as plt
from sklearn import svm

## 1 SVM ##
## 1.1 Example Dataset 1 ##
os.chdir("d:/data/ex6/")

all_data = scipy.io.loadmat("ex6data1.mat")

X = all_data['X']
y = all_data['y']

pos = np.array([X[i] for i in range(len(X)) if y[i] == 1])
neg = np.array([X[i] for i in range(len(X)) if y[i] == 0])

def plotting():
    plt.figure()
    plt.plot(pos[:,0], pos[:,1], 'b+')
    plt.plot(neg[:,0], neg[:,1], 'ro')

def plotBoundary(my_svm, xmin, xmax, ymin, ymax):
    xvals = np.linspace(xmin,xmax,100)
    yvals = np.linspace(ymin,ymax,100)
    zvals = np.zeros((len(xvals), len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[i][j] = float(my_svm.predict(np.array([xvals[i], yvals[j]]).reshape(-1,2)))
    zvals = zvals.transpose()
    u, v = np.meshgrid(xvals, yvals)
    mycontour = plt.contour(xvals, yvals, zvals, [0])

c_svm = svm.SVC(C=1, kernel='linear')
c_svm.fit(X, y.flatten())

predict = c_svm.predict(X)

#plotting()
#plotBoundary(c_svm, 0, 4.5, 1.5, 5)

# C = 100 #
c100_svm = svm.SVC(C=100, kernel='linear')
c100_svm.fit(X, y)

predict = c_svm.predict(X)
#plotting()
#plotBoundary(c100_svm, 0, 4.5, 1.5, 5)

## 1.2 SVM with Gaussian Kernels ##
## 1.2.1 Gaussian Kernel ##

def gaussian(x1, x2, sigma):
    return np.exp(-np.dot((x1-x2).T, (x1-x2)) / (2 * np.square(sigma)))

#print(gaussian(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.)) # 0.32465
    
## 1.2.2 Example Dataset 2 ##
all_data = scipy.io.loadmat("ex6data2.mat")

X = all_data['X']
y = all_data['y']

pos = np.array([X[i] for i in range(len(X)) if y[i] == 1])
neg = np.array([X[i] for i in range(len(X)) if y[i] == 0])

#plotting()

sigma = 0.1
gamma = np.power(sigma, -2)
rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=gamma)
rbf_svm.fit(X, y)

plotting()
plotBoundary(rbf_svm, 0, 1, 0.4, 1.0)

## 1.2.3 Example Datset 3 ##
all_data = scipy.io.loadmat("ex6data3.mat")

X = all_data['X']
y = all_data['y']
x_cv = all_data['Xval']
y_cv = all_data['yval']

pos = np.array([X[i] for i in range(len(X)) if y[i] == 1])
neg = np.array([X[i] for i in range(len(X)) if y[i] == 0])

sigma = 0.1
gamma = np.power(sigma, -2)
rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=gamma)

plotting()
C = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
sigmas = C
best_pair, best_score = [0, 0], 0
for c in C:
    for sm in sigmas:
        gamma = np.power(sm, -2)
        gs_svm = svm.SVC(C=c, kernel='rbf', gamma=gamma)
        gs_svm.fit(X, y.flatten())
        score = gs_svm.score(x_cv, y_cv)
        if score > best_score:
            best_score = score
            best_pair[0], best_pair[1] = c, sm

print(best_score, best_pair)

gs_svm_test = svm.SVC(C=best_pair[0], kernel='rbf', gamma=np.power(best_pair[1], -2))
gs_svm_test.fit(X, y.flatten())
plotting()
plotBoundary(gs_svm_test,-.5,.3,-.8,.6)