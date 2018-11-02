import tensorflow as tf
import numpy as np

## hidden size ##
## seq length ##
## batch_size ##

tf.set_random_seed(777)

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# parameter #
hidden_size = 2
sequence_length = 5
batch_size = 3

# RNN building #
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True,
                                    reuse=tf.AUTO_REUSE)
x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)

#print(x_data)
sess = tf.Session()
outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
print(sess.run(outputs))
sess.close()