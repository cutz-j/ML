### 3-1. multi-class classification ###
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import io
import matplotlib.cm as cm
import random
import scipy.misc

## 1.1. Dataset ##
all_data = io.loadmat("d:/data/ex03/ex3data1.mat")

x_data = all_data['X']
y_data = all_data['y']
y_data = np.array(y_data).reshape(len(y_data), 1)

## 1.2 Visualizaing the data ##
def pixel(row):
    width, height = 20, 20
    square = row.reshape(width, height)
    return square.T

def displayData(indices_to_display = None):
    """
    Function that picks 100 random rows from X, creates a 20x20 image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
    width, height = 20, 20
    nrows, ncols = 10, 10
    if not indices_to_display:
        indices_to_display = random.sample(range(x_data.shape[0]), nrows*ncols)
        
    big_picture = np.zeros((height*nrows,width*ncols))
    
    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol  = 0
        iimg = pixel(x_data[idx])
        big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(6,6))
    img = scipy.misc.toimage( big_picture )
    plt.imshow(img,cmap = cm.Greys_r)
    
#displayData()   

## 1.3 Vectorizing Logistic Regression ##

X = tf.placeholder(dtype=tf.float32, shape=[5000, 400])
Y = tf.placeholder(dtype=tf.float32, shape=[5000, 1]) # not one-hot --> 1~10
Y_one_hot = tf.one_hot(y_data, 10, dtype=tf.float32) # one hot method
Y_one_hot = tf.reshape(Y_one_hot, [-1, 10]) # one-hot 차원 줄이기
W = tf.Variable(tf.random_normal([400, 10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')
ld = 0.001 # L2 Reg 람다 값

# 1.3.2 Vectorizing the gradient #

logit = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logit)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y_one_hot))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.9)
#regularizer = tf.nn.l2_loss(W)
#cost_reg = cost + ld + regularizer
train = optimizer.minimize(cost)

#hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#cost = -tf.reduce_mean(tf.log(hypothesis) * tf.transpose(Y) - (1 - Y) * tf.log(1 - hypothesis))
#regularizer = tf.nn.l2_loss(W)
#cost_reg = cost + ld + regularizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
#train = optimizer.minimize(cost_reg)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

prediction = tf.argmax(hypothesis, axis=1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(8000):
    y_hat, _ , cost_val, w_val, acc = sess.run([hypothesis, train, cost, W, accuracy], feed_dict={X: x_data, Y: y_data})
    if i % 300 == 0:
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(i, cost_val, acc))

sess.close()




    
    
    
    
    
    
    
    
    