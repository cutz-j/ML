import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
tf.reset_default_graph() ## reset

sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

dic_size = len(char2idx)
hidden_size = len(char2idx)
sequence_length = len(sample) - 1
num_classes = len(char2idx)
batch_size = 1
learning_rate = 0.1


X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes) # shape 주의

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
init_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=init_state, dtype=tf.float32)

# FC layer
#X_for_fc = tf.reshape(outputs, [-1, hidden_size])
#outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
#outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

predict = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        cost_val, _ = sess.run([loss, train], 
                           feed_dict={X: x_data, Y: y_data})
        res = sess.run(predict, feed_dict={X: x_data})
        
        res_str = [idx2char[c] for c in np.squeeze(res)]
        if i % 100 == 0:
            print(i, cost_val)
            print(''.join(res_str))
    

























