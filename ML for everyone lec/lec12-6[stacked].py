### stacked RNN ###
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)
tf.reset_default_graph() ## reset

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}
dataX = []
dataY = []

seq_length = 10 # 임의의 sequence length --> 영향 받는 데이터

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i: i + seq_length]
    y_str = sentence[i + 1: i + seq_length + 1]
    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]
    
    dataX.append(x)
    dataY.append(y)
    
# hyper parameter #
learning_rate = 0.1
batch_size = len(dataX) # 데이터 개수
seq_length = 10  # 25 --> sequence (영향받는 데이터 개수) --> 기억개수
hidden_size = len(char_dic) # 25 --> 출력 25 (LSTM에서의 아웃풋)
num_classes = len(char_dic) # 25 --> 출력 개수 (최종 결과물)

## RNN multicell ##
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
cell = tf.contrib.rnn.MultiRNNCell([cell] * 3, state_is_tuple=True)

# tf building #
X = tf.placeholder(dtype=tf.int64, shape=[None, seq_length]) # seq_lengTh만 지정
Y = tf.placeholder(dtype=tf.int64, shape=[None, seq_length])
X_one_hot = tf.one_hot(X, num_classes) # x_one_hot

# 초기화 #
init_state = cell.zero_state(batch_size, dtype=tf.float32)

# cell 통과 #
output, _state = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=init_state, dtype=tf.float32)

## reshape for softmax ##
X_for_softmax = tf.reshape(output, shape=[-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_softmax, num_classes, activation_fn=None)
#softmax_W = tf.Variable([hidden_size, num_classes], dtype=tf.float32, name='softmax_w')
#softmax_b = tf.Variable([num_classes], dtype=tf.float32, name='softmax_b')
#logits = tf.matmul(X_for_softmax, softmax_W) + softmax_b
#hypothesis = tf.nn.softmax(logits)

# output reshape #
output = tf.reshape(outputs, shape=[batch_size, seq_length, hidden_size])

## cost ##
weights = tf.ones([batch_size, seq_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=output, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(seq_loss)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = opt.minimize(mean_loss)



# session #
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    cost_var, _, y_hat = sess.run([mean_loss, train, output],
                                  feed_dict={X: dataX, Y: dataY})
    if i % 100 == 0:
        print(cost_var)
        
        for j, res in enumerate(y_hat):
            index = np.argmax(res, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), 1)
    


results = sess.run(output, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j == 0:
        print(''.join([char_set[t] for t in index]), 1)
    else:
        print(char_set[index[-1]], end='')

sess.close()































