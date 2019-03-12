import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

data=np.loadtxt("../TodayILearned/DA/data/iris_softmax.csv", delimiter=",", dtype=np.float32)
np.random.shuffle(data)

x_data=data[:120, :5]
y_data=data[:120, 5:]
x_test=data[120:, :5]
y_test=data[120:, 5:]

classes=3

X=tf.placeholder(tf.float32, shape=[None, 5])
Y=tf.placeholder(tf.int32, shape=[None, 3])

W=tf.Variable(tf.random_normal([5, classes]), name='Weight')
b=tf.Variable(tf.random_normal([classes]), name='bias')

fX=tf.matmul(X, W)+b
hypothesis=tf.nn.softmax(fX)

cost_e=tf.nn.softmax_cross_entropy_with_logits(logits=fX, labels=y_data)
#cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
cost=tf.reduce_mean(cost_e)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, axis=1)
correct_prediction = tf.equal(prediction, tf.argmax(y_data, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        
        if step%100==0:
            c, acc=sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, c, acc))
            
    predict=sess.run(prediction, feed_dict={X:x_test})
    for p, y in zip(predict, np.argmax(y_test, axis=1)):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))