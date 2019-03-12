### 8-1. Anomaly Detection and Recommender Systems ###
import numpy as np
import scipy.io as io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

### 1. Anomaly Detection ###
all_data = io.loadmat("c:/data/ex8/ex8data1.mat")
X = all_data['X']
x_cv = all_data['Xval']
y_cv = all_data['yval']

def plotData(myX, newFig=False):
    if newFig:
        plt.figure(figsize=(8,6))
    plt.plot(myX[:,0],myX[:,1],'b+')
    plt.xlabel('Latency [ms]',fontsize=16)
    plt.ylabel('Throughput [mb/s]',fontsize=16)
    plt.grid(True)

#plotData(X)

## 1.1 Gasussian distribution ##
## 1.2 Estimating parameters for a Gaussian ##
def gaus(X, mu, sigma):
    dummy = np.zeros_like(X)
    for i in range(dummy.shape[1]):
        #  feature만큼 반복 #
        fore = 1. / np.sqrt(2 * np.pi * sigma[i])
        exp = - (np.square(X[:, i]- mu[i])) / (2 * sigma[i])
        dummy[:, i] = fore *  np.exp(exp)
    return dummy

def gaus_param(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.mean(np.square(X - mu), axis=0)
    return mu, sigma2

mu, sigma = gaus_param(X)
X_gaus = gaus(X, mu, sigma)

## 1.2.1 Visualizing the Gaussian prob contours ##
def plotContours(mymu, mysigma2, newFig=False, useMultivariate = True):
    delta = .5
    myx = np.arange(0, 30, delta)
    myy = np.arange(0, 30, delta)
    meshx, meshy = np.meshgrid(myx, myy)
    coord_list = [ entry.ravel() for entry in (meshx, meshy) ]
    points = np.vstack(coord_list).T
    myz = gaus(points, mymu, mysigma2)
    if newFig: plt.figure(figsize=(6,4))
    cont_levels = [10**exp for exp in range(-20,0,3)]
    plt.contour(meshx, meshy, myz[:, 0].reshape(60, 60), levels=cont_levels)
    plt.title('Gaussian Contours',fontsize=16)


plotData(X, newFig=True)
plotContours(mu, sigma, newFig=False)

#plt.figure()
#plt.hist(X_gaus[:,0])
#plt.show()

## 1.3 Selecting threshold ##
     
def optthreshold(x_cv, y_cv):
    ## grid search ##
    mu, sigma = gaus_param(x_cv)
    x_cv_gaus = gaus(x_cv, mu, sigma)
    nsteps = 1000
    epses = np.linspace(np.min(x_cv_gaus), np.max(x_cv_gaus), nsteps)
    best_eps, best_f1 = 0, 0
    for eps in epses:
        y_hat = np.zeros_like(y_cv)
        idx = np.where(x_cv_gaus < eps)[0]
        idx = np.unique(idx)
        y_hat[idx] = 1
        score = f1_score(y_cv, y_hat)
        if score > best_f1:
            best_f1 = score
            best_eps = eps
    return best_eps, best_f1

eps, f1 = optthreshold(x_cv, y_cv)
idx = np.where(X_gaus < eps)[0]

plotData(X, newFig=True)
plotContours(mu, sigma, newFig=False)
plt.plot(X[idx][:, 0], X[idx][:, 1], 'r+')
plt.show()

mat = io.loadmat("c:/data/ex8/ex8data2.mat")
Xpart2 = mat['X']
ycvpart2 = mat['yval']
Xcvpart2 = mat['Xval']

mu, sigma = gaus_param(Xpart2)
X_gaus2 = gaus(Xpart2, mu, sigma)
cv_gaus = gaus(Xcvpart2, mu, sigma)
eps, f1 = optthreshold(Xcvpart2, ycvpart2)
anoms = [Xpart2[x] for x in range(len(Xpart2)) if np.array(X_gaus2[x] < eps).any()]
## 61개, eps==0.0001 ##