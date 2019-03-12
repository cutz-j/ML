import tensorflow as tf

#X=[1,2,3]
#Y=[1,2,3]

# 매개변수
X=tf.placeholder(tf.float32, shape=[None])
Y=tf.placeholder(tf.float32, shape=[None])

# tf변수
W1 = tf.Variable(tf.random_normal([1]), name="weight1")
b1 = tf.Variable(tf.random_normal([1]), name="bias1")
L1 = X * W1 + b1

W2 = tf.Variable(tf.random_normal([1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")
L2 = L1 * W2 + b2

W3 = tf.Variable(tf.random_normal([1]), name="weight3")
b3 = tf.Variable(tf.random_normal([1]), name="bias3")
hypothesis = L2 * W3 + b3

# 오차 최소제곱
cost=tf.reduce_mean(tf.square(hypothesis-Y))

# train
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, _ = sess.run([cost, train],
                 feed_dict={X:[1,2,3,4,5], Y: [2.1,3.1,4.1,5.1,6.1]})
    if step % 100==0:
        print(step, cost_val)

# Test
print(sess.run(hypothesis, feed_dict={X:[5]}))
        
#for step in range(2001):
#    sess.run