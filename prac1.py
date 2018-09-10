import pandas as pd
import tensorflow as tf
import numpy as np

dfs=pd.read_excel("../BigData-project/2017_price_stock.xlsx").ix[:,::-1]
df_test=pd.read_excel("../BigData-project/2018_price_stock.xlsx").ix[:,::-1]

'''
Simple Regression // price ~ stock
'''
#NA처리
dfs=dfs.dropna()
df_test=df_test.dropna()

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

x_single_data=MinMaxScaler(np.array([dfs.ix[:,0].values]).T)
y_data=MinMaxScaler(np.array([dfs.ix[:,-1].values]).T)

x_single_test=MinMaxScaler(np.array([df_test.ix[:,0].values]).T)
y_test=MinMaxScaler(np.array([df_test.ix[:,-1].values]).T)

X=tf.placeholder(tf.float32, shape=[None, 1])
Y=tf.placeholder(tf.float32, shape=[None, 1])

W=tf.Variable(tf.random_normal([1,1]), name='Weight')
b=tf.Variable(tf.random_normal([1]), name='bias')
hypothesis=tf.matmul(X, W)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        cost_v, hx, _ = sess.run([cost, hypothesis, optimizer], 
                                 feed_dict={X:x_single_data, Y:y_data})
        if step%50==0:
            print("cost: ", cost_v)
            
    # prediction test
    y_hat=sess.run([hypothesis], feed_dict={X:x_single_test})
    for i,j in zip(y_hat[0], y_test):
        print(i,j)

