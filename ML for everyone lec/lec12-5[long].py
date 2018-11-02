import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
tf.reset_default_graph() ## reset

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}
dataX = []
dataY = []


# hyper parameter #
seq_length = 10
dic_size = len(char_dic)
hidden_size = len(char_dic)
num_classes = len(char_dic)
batch_size = len(dataX)
learning_rate = 0.1


for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i: i + seq_length]
    y_str = sentence[i + 1: i + seq_length + 1]
    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]
    
    dataX.append(x)
    dataY.append(y)
    

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])
X_one_hot = tf.one_hot(X, num_classes) # shape 주의

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
init_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=init_state, dtype=tf.float32)

# FC layer
#X_for_fc = tf.reshape(outputs, [-1, hidden_size])
#outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
#outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, seq_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(400):
        cost_val, _, res = sess.run([loss, train, outputs], 
                           feed_dict={X: dataX, Y: dataY})
        # 결과
        for j, result in enumerate(res):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), 1)
        
    results = sess.run(outputs, feed_dict={X: dataX})
    for j, res in enumerate(results):
        index = np.argmax(res, axis=1)
        if j == 0:
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')
    

























