'''
tf 이용하여 Linear Regression 구현
'''
import numpy as np
import tensorflow as tf

# 학습 데이터 뽑기
data = open("ex1data1.txt" , "r", encoding="utf-8")
data = data.readlines()
data_split = []
for i in range(len(data)):
    data_split.append(data[i].strip('\n').split(','))
data = np.array(data_split, dtype=np.float32)

# 학습 데이터와 예측 데이터 나누기
x_data, y_data = data[:91,0], data[:91,1]
x_predict, y_predict = data[91:,0], data[91:,1]

# hypothesis // NN
X=tf.placeholder(tf.float32, shape=[None])
Y=tf.placeholder(tf.float32, shape=[None])

W1 = tf.Variable(tf.random_normal([1]), name="weight1")
b1 = tf.Variable(tf.random_normal([1]), name="bias1")
L1 = X * W1 + b1

W2 = tf.Variable(tf.random_normal([1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")
L2 = L1 * W2 + b2

W3 = tf.Variable(tf.random_normal([1]), name="weight3")
b3 = tf.Variable(tf.random_normal([1]), name="bias3")
hypothesis = L2 * W3 + b3

# cost
learning_rate = 0.001
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

# 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(5001):
    cost_value,  _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print(cost_value)
        
# 예측
print(sess.run(hypothesis, feed_dict={X:x_data}))
print(y_data)