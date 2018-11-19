### programming Exercise 5: Reg Linear Regression and Bias vs Variance ###
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(777)

### 1 Regularized Linear Regression ###
all_data = scipy.io.loadmat("d:/data/ex5/ex5data1.mat")
X = all_data['X'] # shape(12, 1)
X_test = all_data['Xtest'] # (21, 1)
X_cv = all_data['Xval'] # (21, 1)
y = all_data['y'] # (12, 1)
ytest = all_data['ytest'] # (21, 1)
y_cv = all_data['yval'] # (21, 1)

## 1.1 Visualizing the dataset ##
def plot():
    plt.figure()
    plt.plot(X[:,1:], y, "rx")
    plt.show()

## 1.2 Regularized linear regression cost function ##
X = np.insert(X, 0, 1, axis=1) # shape(12, 2)
X_cv = np.insert(X_cv, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)
init_w = np.ones(shape=(2,1))

def hypothesis(X, W):
    return np.matmul(X, W)

def cost(X, W, y, myld):
    m = len(X)
    return (1 / 2) * np.mean(np.square((hypothesis(X, W) - y))) + (myld * (1 / (2 * m)) * (np.sum(np.square(W[1:]))))
     
#print(cost(X, init_w, y, 1.)) # 303.993

## 1.3 Regularized linear regression gradient ##
def gradientDescent(X, W, y, myld, learning_rate=0.001, iteration=1000):
    m = len(X)
    for i in range(iteration):
        grad = (1. / m) * np.sum((hypothesis(X, W) - y) * X, axis=0)
        grad = grad.reshape(len(W), 1)
        reg = (myld / m) * W
        reg[0] = 0
        reg.reshape(len(W), 1)
        W = W - learning_rate * (grad + reg)
    return W

new_W = gradientDescent(X, init_w, y, 0, 1.49e-1, 1000) # [-15.30, 598.25]

## 1.4 Fitting linear regression
#plt.figure()
#plt.plot(X[:,1], y, "rx")
#plt.plot(X[:,1], hypothesis(X, new_W), "b-")
#plt.show()
#    
    
### 2 Bias-variance ###   
## 2.1 Learning curves ##
def learning_curve(X, X_cv, y, y_cv, W, myld):
    m = len(X)
    m_list, x_cost, cv_cost = [], [], []
    for i in range(1, m+1):
        train_set = X[:i, :]
        y_set = y[:i, :]
        m_list.append(len(train_set))
        new_w = gradientDescent(train_set, W, y_set, myld=0)
        x_cost.append(cost(train_set, new_w, y_set, myld=0))
        cv_cost.append(cost(X_cv, new_w, y_cv, myld=0))
    plt.figure()
    plt.plot(m_list, x_cost, 'r-')
    plt.plot(m_list, cv_cost, 'b-')
    plt.show()
    
#learning_curve(X, X_cv, y, y_cv, init_w, myld=0)

### 3 polynomial regression ###
def poly(X, p):
    ## p만큼 polyt nomial feature 추가 ##    
    new_X = X.copy()
    for i in range(2, p+1):
        new_X = np.insert(new_X, new_X.shape[1], np.power(X[:,1], i), axis=1)
    return new_X

X_poly = poly(X, 5)        

## 3-1 Learning Polynomial Regression ##   
def normalize(X):
    Xnorm = X.copy()
    stored_feature_means = np.mean(Xnorm, axis=0) #column-by-column
    Xnorm[:,1:] = Xnorm[:,1:] - stored_feature_means[1:]
    stored_feature_stds = np.std(Xnorm, axis=0, ddof=1)
    Xnorm[:,1:] = Xnorm[:,1:] / stored_feature_stds[1:]
    return Xnorm, stored_feature_means, stored_feature_stds

X_norm, means, stds = normalize(X_poly) # shape(12, 9)
init_w = np.ones(shape=(X_norm.shape[1], 1))
new_w = gradientDescent(X_norm, init_w, y, myld=0)
    
#def plotFit(W, means, stds):
#    n_points_to_plot = 50
#    xvals = np.linspace(-55, 55, n_points_to_plot)
#    xmat = np.ones((n_points_to_plot, 1))
#    xmat = np.insert(xmat, xmat.shape[1], xvals.T, axis=1)
#    xmat = poly(xmat, len(W)-2)
#    xmat[:,1:] = xmat[:,1:] - means[1:]
#    xmat[:,1:] = xmat[:,1:] / stds[1:]
#    plot()
#    plt.plot(xvals, hypothesis(xmat, W),'b--')
#
#plotFit(new_w, means, stds)

def poly_learning_curve(X, X_cv, y, y_cv, W, myld):
    m = len(X)
    m_list, x_cost, cv_cost = [], [], []
    for i in range(1, m+1):
        train_set = X[:i, :]
        y_set = y[:i]
        m_list.append(len(train_set))
        new_w = gradientDescent(train_set, W, y_set, myld)
        x_cost.append(cost(train_set, new_w, y_set, myld))
        cv_cost.append(cost(X_cv, new_w, y_cv, myld))
    plt.figure()
    plt.plot(m_list, x_cost, 'r-')
    plt.plot(m_list, cv_cost, 'b-')
    plt.show()

X_cv_norm, means_cv, stds_cv = normalize(poly(X_cv, 5))
poly_learning_curve(X_norm, X_cv_norm, y, y_cv, init_w, myld=1)

lambdas = np.linspace(0,5,20)
err_train, err_val = [], []
for ld in lambdas:
    new_train = poly(X, 5)
    x_norm, _, _ = normalize(new_train)
    new_cv = poly(X_cv, 5)
    x_cv_norm, _, _ = normalize(new_cv)
    new_w = gradientDescent(x_norm, init_w, y, ld, learning_rate=0.003, iteration=1000)
    err_train.append(cost(x_norm, new_w, y, ld))
    err_val.append(cost(x_cv_norm, new_w, y_cv, ld))

plt.figure()
plt.plot(lambdas, err_train, 'r-')
plt.plot(lambdas, err_val, 'b-')
plt.show()
