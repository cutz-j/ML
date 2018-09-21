import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps
from sklearn import preprocessing as pp


dfs=pd.read_excel("../BigData-project/2017_price_stock.xlsx").ix[:,::-1]
df_test=pd.read_excel("../BigData-project/2018_price_stock.xlsx").ix[:,::-1]

'''
Simple Regression // price ~ stock
'''

'''
all data 전처리
이상치 처리(noise)
정규성 검정
정규화 + 일반화 -> 학습+검정데이터 모두
'''
dfs=dfs.dropna()
df_test=df_test.dropna()

rs=pp.RobustScaler()
rs.fit(dfs, df_test)
dfs_rs=rs.transform(dfs)
df_test_rs=rs.transform(df_test)
print(dfs_rs, df_test_rs)

#NA처리


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def standard(data):
    return (data[:]-data.mean())/data[:].std()

x_single_data=MinMaxScaler(np.array([dfs.ix[:,-1].values]).T)
y_data=MinMaxScaler(np.array([dfs.ix[:,-1].values]).T)

x_single_tune= MinMaxScaler(np.array([dfs.ix[65:,0].values]).T)
y_tune=MinMaxScaler(np.array([dfs.ix[65:,0].values]).T)

x_single_test=standard(np.array([df_test.ix[:,0].values]).T)
y_test=standard(np.array([df_test.ix[:,-1].values]).T)

X=tf.placeholder(tf.float32, shape=[None, 1])
Y=tf.placeholder(tf.float32, shape=[None, 1])

W=tf.Variable(tf.random_normal([1,1]), name='Weight')
b=tf.Variable(tf.random_normal([1]), name='bias')
hypothesis=tf.matmul(X, W)+b

l2reg=0.001*tf.reduce_sum(tf.square(W))

cost=tf.reduce_mean(tf.square(hypothesis-Y))+l2reg
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(501):
        cost_v, hx, _ = sess.run([cost, hypothesis, optimizer], 
                                 feed_dict={X:x_single_data, Y:y_data})
        if step%50==0:
            print("cost: ", cost_v)
            
    # prediction test
    y_hat=sess.run([hypothesis], feed_dict={X:x_single_test})
    for i,j in zip(y_hat[0], y_test):
        print(i,j)

