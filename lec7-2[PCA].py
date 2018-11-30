### Assignment 7-2. Pricipan Component Analysis ###
import scipy.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import randomized_svd
import matplotlib.cm as cm
import scipy

## 2.1 Example Dataset ##
data1 = io.loadmat("d:/data/ex7/ex7data1.mat")
X = data1['X'] # shape(50, 2)
#plt.figure()
#plt.plot(X[:,0], X[:,1], 'bo')

# standardzation #
ss = StandardScaler()
X_scale = ss.fit_transform(X)

# square #

def svd(X):
    cov_matrix = X.T.dot(X) / X.shape[0] # shape(2, 2)
    U, S, V = np.linalg.svd(cov_matrix)
    return U, S, V

U, S, V = svd(X_scale)

plt.figure()
plt.plot(X[:,0], X[:,1], 'bo')
#U_reverse = ss.inverse_transform(U)
mean = np.mean(X, axis=0)
#plt.plot([mean[0], mean[0] + 1.5 * S[0] * U[0, 0]], [mean[1], mean[1] + 1.5 * S[0] * U[0, 1]], 'r-')
#plt.plot([mean[0], mean[0] + 1.5 * S[1] * U[1, 0]], [mean[1], mean[1] + 1.5 * S[1] * U[1, 1]], 'g-')
#plt.show()

print(U[0]) # [-0.707, -0.707]

## 2.3 Dimensionality Reduction with PCA ##
# 2.3.1 Projecting the data onto the principal components #
def reduce(X, U, K):
    ## Unitary matrix의 K 컬럼만큼 행렬연산 ##
    return np.dot(X, U[:,:K])

K = 1
z = reduce(X_scale, U, K).reshape(-1,K)
#print(z[0]) # 1.496

# 2.3.2 Reconstructing an approximation of the data #
#svd = TruncatedSVD(n_components=1)
#svd.fit_transform(X_scale)
#svd.explained_variance_ratio_

def recover(z, U, K):
    return np.dot(z, U[:,:K].T)

x_recover = recover(z, U, K)
#print(x_recover[0])  # [-1.05805279, -1.05805279]

## 2.4 Face Image Dataset ##
face = io.loadmat("d:/data/ex7/ex7faces.mat")
X = face['X'] # shape(5000, 1024)
#plt.imshow(X)
def getDatumImg(row):
    width, height = 32, 32
    square = row.reshape(width,height)
    return square.T

def displayData(myX, mynrows = 10, myncols = 10):
    width, height = 32, 32
    nrows, ncols = mynrows, myncols

    big_picture = np.zeros((height*nrows,width*ncols))
    
    irow, icol = 0, 0
    for idx in range(nrows*ncols):
        if icol == ncols:
            irow += 1
            icol  = 0
        iimg = getDatumImg(myX[idx])
        big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(10,10))
    img = scipy.misc.toimage( big_picture )
    plt.imshow(img,cmap = cm.Greys_r)
#displayData(X)
    
# scaling #
X_scale = ss.fit_transform(X)

# 2.4.1 PCA on Faces #
svd_sk = TruncatedSVD(n_components=36, n_iter=5)
X_svd = svd_sk.fit_transform(X_scale)
U1, S1, V1 = svd(X_scale)
X_rec = svd_sk.inverse_transform(X_svd)

#displayData(U[:,:36].T, mynrows=6, myncols=6)
z = reduce(X_scale, U, K=36)
x_rec = recover(z, U, K=36)
U, S, V = randomized_svd(X_scale, n_components=36)
#displayData(x_rec)