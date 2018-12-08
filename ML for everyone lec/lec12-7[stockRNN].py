### stock daily RNN-LSTM ###
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import rnn

tf.set_random_seed(777)
tf.reset_default_graph()

## parameter ##
seq_length = 7
data_dim = 5
output_dim = 1
hidden_size = 30
learning_rate = 0.03
iteration = 5000
is_training = True
keep_prob = tf.placeholder(dtype=tf.float32)
l2norm = 0.0001

### 데이터 전처리 ###
stock_data = np.loadtxt("c:/data/prac/stock_daily.csv", delimiter=',', skiprows=1)
stock_data = stock_data[::-1] # shape (732, 5)

# train scaling #
mm1 = MinMaxScaler()
stock_data_x = mm1.fit_transform(stock_data[:, :-1])
stock_data_y = mm1.fit_transform(stock_data[:, [-1]])
stock_data = np.concatenate((stock_data_x, stock_data_y), axis=1)

## split ## --> 시계열(시간순)
train_size = int(len(stock_data) * 0.7)
train_set = stock_data[:train_size, :] # shape(512, 5)
test_set = stock_data[train_size:, :] # test(220, 5)


# RNN data building #
def build(time_series, seq_length):
    x_data = []
    y_data = []
    for i in range(0, len(time_series) - seq_length):
        x_tmp = time_series[i: i + seq_length, :]
        y_tmp = time_series[i + seq_length, [-1]]
        x_data.append(x_tmp)
        y_data.append(y_tmp)
    return np.array(x_data), np.array(y_data)

x_train, y_train = build(train_set, seq_length)
x_test, y_test = build(test_set, seq_length)

## RNN building ##
# cell #
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True, activation=tf.tanh)
    return cell

#cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.tanh)
cell = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
#
X = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, data_dim])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
#
## 초기화 #
output, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) 
Y_pred = tf.contrib.layers.fully_connected(output[:, -1], output_dim, activation_fn=None) # last cell output --> 15일 뒤

init = tf.contrib.layers.xavier_initializer(seed=77)
W1 = tf.Variable(init([1, 100]), name='weight1')
b1 = tf.Variable(init([100]), name='bias1')
layer1 = tf.matmul(Y_pred, W1) + b1
l1 = tf.contrib.layers.batch_norm(layer1, center=True, scale=True,
                                  is_training=is_training)
L1 = tf.nn.relu(l1, name='relu1')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(init([100, 1]), name='weight2')
b2 = tf.Variable(init([1]), name='bias2')
hypothesis = tf.matmul(L1, W2) + b2

## tf.trainable --> l2 norm ##
var = tf.trainable_variables()
l2reg = tf.add_n([tf.nn.l2_loss(v) for v in var if 'bias' not in v.name]) * l2norm

# cost #
cost = tf.reduce_sum(tf.square(Y_pred - y)) # sum of sq --> 수치 예측이기 때문에 sq loss가 필요 없다.
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch_norm
with tf.control_dependencies(update_ops):
    train = opt.minimize(cost)

# MSE # --> mean squared error
targets= tf.placeholder(tf.float32, [None, 1])
predicts = tf.placeholder(tf.float32, [None, 1])
MSE = tf.sqrt(tf.reduce_mean(tf.square(predicts - targets)))

## session ##
# training#

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
for i in range(iteration):
    cost_val, _, out= sess.run([cost, train, output], feed_dict={X: x_train, y: y_train, keep_prob: 0.5})
    if i % 300 == 0:
        print(cost_val)

# predict #
is_training = False
y_hat_train = sess.run(Y_pred, feed_dict={X: x_train, keep_prob: 1.0})
y_hat = sess.run(Y_pred, feed_dict={X: x_test, keep_prob: 1.0})
RMSE_train = sess.run(MSE, feed_dict={targets: y_train, predicts: y_hat_train, keep_prob: 1.0})
RMSE = sess.run(MSE, feed_dict={targets: y_test, predicts: y_hat, keep_prob: 1.0})
print("RMSE_train: ", RMSE_train)
print("RMSE: ", RMSE)

sess.close()

y_hat = mm1.inverse_transform(y_hat)
y_test = mm1.inverse_transform(y_test)

plt.figure()
plt.plot(y_train, 'r-')
plt.plot(y_hat_train, 'b-')
plt.show()

plt.figure()
plt.plot(y_test, 'r-')
plt.plot(y_hat, 'b-')
plt.show()








