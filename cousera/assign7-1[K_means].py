### Assignment 7-1. K-means Clustering ###
import numpy as np
import scipy.io as io
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.misc as misc

## 1.1 Implementing K-means ##
data1 = io.loadmat("d:/data/ex7/ex7data2.mat")
X = data1['X']

# init centriods #
K = 3
init_centroids = np.array([[3.,3.], [6.,2.], [8.,5.]])

# visual #
def visual(X, init):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], color='b', marker='o')
    plt.scatter(init[:, 0], init[:, 1], color='r', marker='x')
    plt.show()
    
#visual(X, init_centroids)

# 1.1.1 Finding closest centroids #

def distance(x1, x2):
    ## x1과 x2의 각각의 거리를 구하는 함수 ##
    return np.sqrt(np.sum(np.square(x2 - x1)))

def find_cluster(X, init):
    ## 가장 가까운 centroid에 clustering을 assign ##
    cluster = []
    for i in range(len(X)):
        best, idx = 100000000, 0
        for j in range(len(init)):
            dist = distance(init[j], X[i])
            if dist < best:
                best = dist
                idx = j
        cluster.append(idx)
    return np.array(cluster)

init_cluster = find_cluster(X, init_centroids)
#print(init_cluster)

# visual #
def visual_cluster(X, init, cluster):
    plt.figure()
    color = ['g', 'b', 'y']
    for i in range(len(cluster)):
        plt.scatter(X[i,0], X[i,1], color=color[cluster[i]], marker='o')
    plt.scatter(init[:, 0], init[:, 1], color='r', marker='x')
    plt.show()
    
#visual_cluster(X, init_centroids, init_cluster)

# 1.1.2 Computing centroid means #
def centroid_means(X, init, cluster):
    ## 각 클러스터의 중앙값과 클러스터 구성원간의 오차 에러 ##
    K = len(init)
    C = []
    for i in range(K):
        ck = X[cluster==i]
        C.append(np.mean(ck, axis=0))
    return np.array(C)

means = centroid_means(X, init_centroids, init_cluster)
#print(means) # [2,79, 4.22, 5.36]

## 1.2 K-means on example dataset ##
def K_means(X, init, K=3, n_iter=10):
    ## K_means를 실제 실행하는 학습 --> 학습 ##
    # 1단계: 각 구성원 지정, 2단계: 각 구성원 평균으로 centroids 초기화 #
    hist = []
    for i in range(n_iter):
        cluster = find_cluster(X, init)
        new_centroid = centroid_means(X, init, cluster)
        hist.append(new_centroid)
        init = new_centroid
    return cluster, np.array(hist)

idx, hist = K_means(X, init_centroids, n_iter=100)

def visual_cluster_learn(X, cluster, hist):
    plt.figure()
    color = ['g', 'b', 'y']
    for i in range(len(cluster)):
        plt.scatter(X[i,0], X[i,1], color=color[cluster[i]], marker='o')
    plt.plot(hist[:,0][:, 0], hist[:,0][:, 1], 'r-x')
    plt.plot(hist[:,1][:, 0], hist[:,1][:, 1], 'g-x')
    plt.plot(hist[:,2][:, 0], hist[:,2][:, 1], 'b-x')
    plt.show()

#visual_cluster_learn(X, idx, hist)

## 1.3 Random init ##
def random_choice(X, k):
    ## random_Shuffle_X ##
    return np.array(random.choices(X, k=k))

#idx, hist = K_means(X, random_choice(X), n_iter=20)
#visual_cluster_learn(X, idx, hist)

## 1.4 Image compression with K-means ##
# 1.4.1 K-means on pixes #
pic = misc.imread("d:/data/ex7/bird_small.png")
#plt.imshow(pic)

# scaling #
pic = pic / 255

# pixel을 한 행으로 #
pic = pic.reshape(-1, 3) # shape(16384, 3)

K = 16
idx, hist = K_means(pic, random_choice(pic, K), K=K, n_iter=10)

def visual_cluster_pic(X, cluster, hist, K):
    plt.figure()
    color = ['g', 'b', 'y']
    for i in range(len(cluster)):
        plt.scatter(X[i,0], X[i,1], color=color[cluster[i]], marker='o')
    for k in range(K):
        plt.plot(hist[:,k][:, 0], hist[:,k][:, 1], 'r-x')
    plt.show()

rgb = centroid_means(pic, hist[-1], idx) * 255
rgb = rgb.astype(np.int32)

dummy = np.zeros_like(pic, dtype=np.int32)
for i in range(len(dummy)):
    dummy[i] = rgb[idx[i]]

plt.figure()
plt.imshow(pic.reshape(128, 128, 3))
plt.figure()
plt.imshow(dummy.reshape(128, 128, 3))
