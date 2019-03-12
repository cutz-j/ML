### Assignment 2-1 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.preprocessing as pp

tf.set_random_seed(777)

## 변수 선언 ##
all_data = pd.read_csv("d:/data/ex02/ex2data1.txt", sep=',', header=None)
x_data = all_data.iloc[:,:2]     # (100, 2)
y_data = np.array(all_data[2])   # (100, 1)
m = y_data.size # 100
y_data = y_data.reshape(m, 1)

iteration = 5000
learning_rate = 0.02

scale = pp.StandardScaler()
x_scale = scale.fit_transform(x_data)

## 1.1 Visualizing the data ##
admit = all_data.iloc[:,:2][all_data[2]==1]
notadmit = all_data.iloc[:,:2][all_data[2]==0]

def plotting():
    plt.figure(figsize=(10,6))
    plt.plot(admit.iloc[:,0], admit.iloc[:,1], 'ro')
    plt.plot(notadmit.iloc[:,0], notadmit.iloc[:,1], 'bx')
    plt.legend()
    plt.show()
    plt.grid(True)
    
## 1.2 Implementation ##
## 1.2.1 sigmoid ##
# tf building #
X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

#W = tf.Variable(tf.random_normal([2, 1]), name='weight')
#b = tf.Variable(tf.random_normal([1], name='bias'))

init = tf.contrib.layers.xavier_initializer(seed=77)
W = tf.Variable(init(shape=[2,1]))
b = tf.Variable(init(shape=[1]))
hypothesis = tf.nn.sigmoid(tf.matmul(X, W) + b)

## 1.2.2 Cost function and gradient ##
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## 1.2.3 Learning parameters ##
for i in range(iteration):
    y_hat, cost_val, _, w_val, b_val = sess.run([hypothesis, cost, train, W, b], feed_dict={X: x_scale, y: y_data})
    if i % 100 == 0:
        print("cost: ", cost_val)

predict = sess.run(hypothesis, feed_dict={X: scale.transform([[45., 75.,]])})
print(predict) # 0.59%

## 1.2.3 Decision boundary ##
boundary_xs = np.array([np.min(x_data[0]), np.max(x_data[0])])
boundary_ys = (-1. / w_val[1][0]) * (b_val[0] + w_val[0][0] * boundary_xs)
plotting()
plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
plt.legend()

## 1.2.4 Evaluating logistic regression ##
#test = np.array([1., 45, 85])
#answer = hypothesis(theta, test)
#print(answer) #0.7762915904112411 --> admit 확률이 0.776

## predict ##
#def prediction(theta, X):
#    return hypothesis(theta, X) >= 0.5
#
#theta = theta.reshape(3,1)
#admit = np.insert(np.array(admit), 0, 1, axis=1)
#notadmit = np.insert(np.array(notadmit), 0, 1, axis=1)
#p_correct = float(np.sum(prediction(theta, admit)))
#n_correct = float(np.sum(np.invert(prediction(theta, notadmit))))
#tot = len(admit) + len(notadmit)
#correct = float((p_correct + n_correct) / tot)
#print(correct) # 0.89 // 총 100개의 데이터 중 89개 정답 *(학습 데이터에서)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


