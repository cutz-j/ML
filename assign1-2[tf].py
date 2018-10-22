# lec4[Multiple feature Linear Regression]
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## function ##
def standard(x):
    global memoryList
    memoryList.append((np.mean(x), np.std(x)))
    return (x - np.mean(x)) / np.std(x)

def rollback(x, num):
    mean, std = memoryList[num][0], memoryList[num][1]
    return x * std + mean

## Variable ##
data = pd.read_csv("d:/data/ex01/ex1data2.txt", encoding='utf-8', sep=',', header=None)
memoryList = []
m = len(data) # 46 --> data 개수
x1_org = data.loc[:, 0] # x1축
x1_scale = standard(x1_org) # x1 스케일링
x2_org = data.loc[:, 1] # x2축
x2_scale = standard(x2_org) # x2 스케일링
y = data.iloc[:, -1] # y축 (가격)
y = standard(y) # reshape 중요
learning_rate = 0.01
iteration = 1500

## x, y placeholder ##
x1= tf.placeholder(tf.float32) 
x2= tf.placeholder(tf.float32)
Y= tf.placeholder(tf.float32)

## W, b, 가설설정 --> H(x) = Wx + b ##
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = x1 * w1 + x2 * w2 + b

## cost와 gradient Descent ##
cost = tf.reduce_mean(tf.square(hypothesis-Y)) # 오차 --> (h(x) - y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

## 실행 단계 ##
train = optimizer.minimize(cost)

sess = tf.Session() # session open
sess.run(tf.global_variables_initializer()) # 변수 initializer

costList = []
costdf = pd.DataFrame(costList)

for step in range(iteration):
    cost_val, hy_val, _, w1_val, w2_val, b_val = sess.run([cost, hypothesis, train, w1, w2, b], feed_dict={x1: x1_scale, x2: x2_scale, Y:y})
    costList.append(cost_val)
costdf[learning_rate] = costList
    

## cost 그래프 ##
plt.plot(costdf[learning_rate], 'r-')
plt.show()

## fitted-line 그래프 ## 3차원을 2차원에 그리다보니, 그래프 모양이 여려 겹치는 모습
X = pd.DataFrame([x1_scale,x2_scale]).T
plt.plot(X, y, 'rx', markersize=10)
plt.plot(X[0], hy_val, 'b-')
plt.show()

predict = [(1650.-memoryList[0][0])/memoryList[0][1], (3-memoryList[1][0])/memoryList[1][1]]
y_hat = rollback(predict[0] * w1_val + predict[1] * w2_val + b_val, 2)
y_hat

sess.close()
























