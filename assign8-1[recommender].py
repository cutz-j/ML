### assign 8-2. Recommender System ###
import numpy as np
import pandas as pd
import scipy.io as io
import matplotlib.pyplot as plt
from scipy import optimize

## 2.1 movie ratings dataset ##
data = io.loadmat("c:/data/ex8/ex8_movies.mat")
Y = data['Y'] # 평가 one-hot -->  shape(1682, 943)
R = data['R'] # 평점

toystory_avg = np.mean(Y[:, 0][R[:, 0] == 1])
print(toystory_avg) # 3.61

## 2.2 Collaborative filtering learning algorithm ##
## 2.2.1 Collaborative filtering cost function ##
U = Y.shape[0]
M = Y.shape[1]

data_param = io.loadmat("c:/data/ex8/ex8_movieParams.mat")
X = data_param['X'] # shape(1682, 10)
W = data_param['Theta'] # shape(943, 10)
nu = 4; nm = 5; nf = 3
X = X[:nm,:nf]
W = W[:nu,:nf]
Y = Y[:nm,:nu]
R = R[:nm,:nu]

def cost(X, W, Y, ld=0):
    ## cost 구하는 함수 ##
    cost_val = 1/2 * np.sum(np.square((np.dot(X, W.T) * R) - Y))
    reg = (ld / 2.) * np.sum(np.square(W)) + (ld / 2.) * np.sum(np.square(X))
    return cost_val + reg
    
print(cost(X, W, Y))

## 2.2.2 Collaborative filtering gradient ##
def gradientDescent(X, W, Y, learning_rate=0.001, ld=0, iteration=100):    
    ## 교대 자승 학습 ##
    for k in range(iteration):
        X_new = X - learning_rate * np.dot(np.matmul(X, W.T) * R, W)
        W_new = W - learning_rate * np.dot((np.matmul(X, W.T) * R).T, X)
        reg_x = ld * X
        reg_w = ld * W
        X_new += reg_x
        W_new += reg_w
        if k % 10 == 0:
            print(cost(X_new, W_new, Y))
        X = X_new.copy()
        W = W_new.copy()
    return X, W

X_final, W_final = gradientDescent(X, W, Y)
    
## 2.3.1 Recommenations ##
movies = []
file = open("c:/data/ex8/movie_ids.txt", 'rt', encoding="ISO-8859-1") 
for line in file:
    movies.append(' '.join(line.strip('\n').split(' ')[1:]))
file.close()

datafile = 'c:/data/ex8/ex8_movies.mat'
mat = io.loadmat( datafile )
Y = mat['Y']
R = mat['R']
# We'll use 10 features
nf = 10

my_ratings = np.zeros((1682,1))
my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

myR_row = my_ratings > 0
Y = np.hstack((Y,my_ratings))
R = np.hstack((R,myR_row))
nm, nu = Y.shape

def normalizeRatings(myY, myR):
    # The mean is only counting movies that were rated
    Ymean = np.sum(myY,axis=1)/np.sum(myR,axis=1)
    Ymean = Ymean.reshape((Ymean.shape[0],1))
    return myY-Ymean, Ymean

def flattenParams(myX, myTheta):
    return np.concatenate((myX.flatten(),myTheta.flatten()))

# A utility function to re-shape the X and Theta will probably come in handy
def reshapeParams(flattened_XandTheta, mynm, mynu, mynf):
    assert flattened_XandTheta.shape[0] == int(nm*nf+nu*nf)
    reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm,mynf))
    reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu,mynf))
    return reX, reTheta

Ynorm, Ymean = normalizeRatings(Y,R)
X = np.random.rand(nm,nf)
Theta = np.random.rand(nu,nf)
myflat = flattenParams(X, Theta)

mylambda = 10.

def cofiCostFunc(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
    myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
    term1 = myX.dot(myTheta.T)
    term1 = np.multiply(term1,myR)
    cost = 0.5 * np.sum( np.square(term1-myY) )
    
    # Regularization stuff
    cost += (mylambda/2.) * np.sum(np.square(myTheta))
    cost += (mylambda/2.) * np.sum(np.square(myX))
    return cost

def cofiGrad(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
    
    # Unfold the X and Theta matrices from the flattened params
    myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
    term1 = myX.dot(myTheta.T)
    term1 = np.multiply(term1,myR)
    term1 -= myY
    Xgrad = term1.dot(myTheta)
    
    Thetagrad = term1.T.dot(myX)

    # Regularization stuff
    Xgrad += mylambda * myX
    Thetagrad += mylambda * myTheta
    
    return flattenParams(Xgrad, Thetagrad)

def checkGradient(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
    myeps = 0.0001
    nparams = len(myparams)
    epsvec = np.zeros(nparams)
    # These are my implemented gradient solutions
    mygrads = cofiGrad(myparams,myY,myR,mynu,mynm,mynf,mylambda)
    for i in range(10):
        idx = np.random.randint(0,nparams)
        epsvec[idx] = myeps
        loss1 = cofiCostFunc(myparams-epsvec,myY,myR,mynu,mynm,mynf,mylambda)
        loss2 = cofiCostFunc(myparams+epsvec,myY,myR,mynu,mynm,mynf,mylambda)
        mygrad = (loss2 - loss1) / (2*myeps)
        epsvec[idx] = 0
        print ('%0.15f \t %0.15f \t %0.15f' % \
        (mygrad, mygrads[idx],mygrad - mygrads[idx]))

result = optimize.fmin_cg(cofiCostFunc, x0=myflat, fprime=cofiGrad,
                               args=(Y,R,nu,nm,nf,mylambda),
                                maxiter=50,disp=True,full_output=True)

resX, resTheta = reshapeParams(result[0], nm, nu, nf)
prediction_matrix = resX.dot(resTheta.T)
my_predictions = prediction_matrix[:,-1] + Ymean.flatten()
pred_idxs_sorted = np.argsort(my_predictions)
pred_idxs_sorted[:] = pred_idxs_sorted[::-1]
print ("Top recommendations for you:")
for i in range(10):
    print ('Predicting rating %0.1f for movie %s.' % \
    (my_predictions[pred_idxs_sorted[i]],movies[pred_idxs_sorted[i]]))
    
print ("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print ('Rated %d for movie %s.' % (my_ratings[i],movies[i]))