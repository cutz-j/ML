import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:-250, 0:-1]
y_data = xy[:-250, [-1]]

x_test= xy[-250:, 0:-1]
y_test= xy[-250:, [-1]]

batch_size = 20
training_epochs = 20
print(x_data.shape, y_data.shape)

tf

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

## batch train ##
#min_after_dequeue = 10000
#capacity = min_after_dequeue + 3 * batch_size
#train_x_batch, train_y_batch = tf.train.shuffle_batch([xy[:, :-1], xy[-1]], batch_size=batch_size,
#                                                      capacity=capacity, min_after_dequeue=min_after_dequeue,
#                                                      allow_smaller_final_batch=False, seed=77)

train_x_batch, train_y_batch = tf.train.batch([x_data, y_data], batch_size=batch_size)

# Launch graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        cost_sum = 0
        total_batch = int(len(x_data) / batch_size) # 5
        for i in range(total_batch):
            x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})
            cost_sum += cost_val / total_batch 
        print(cost_sum)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_test, Y: y_test})
    print("\nAccuracy: ", a)
    
    coord.request_stop()
    coord.join(threads)