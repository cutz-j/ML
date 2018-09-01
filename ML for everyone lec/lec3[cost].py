import tensorflow as tf
import matplotlib.pyplot as plt

x=[1,2,3]
y=[1,2,3]

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#W = tf.placeholder(tf.float32)
W= tf.Variable(tf.random_normal([1]), name='weight')
hypothesis=W*X

cost=tf.reduce_mean(tf.square(hypothesis-Y))



#W_val=[]
#cost_val=[]
#for i in range(-30, 50):
#    feed_W=i*0.1
#    curr_cost, curr_W=sess.run([cost, W], feed_dict={W: feed_W})
#    W_val.append(curr_W)
#    cost_val.append(curr_cost)
#    

learning_rate=0.1
gradient=tf.reduce_mean((W*X-Y)*X)
descent=W-learning_rate * gradient
update=W.assign(descent)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x, Y: y})
    print(step, sess.run(cost, feed_dict={X: x, Y: y}), sess.run(W))


#plt.plot(W_val, cost_val)
#plt.show()

X=[1,2,3]
Y=[1,2,3]

W = tf.Variable(-3.0)

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)


# 비
X=[1,2,3]
Y=[1,2,3]

W=tf.Variable(5.)
hypothesis=X*W
gradient=tf.reduce_mean((W*X-Y)*X)*2

cost=tf.reduce_mean(tf.square(hypothesis-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Get gradients
gvs=optimizer.compute_gradients(cost, [W])
# Apply grad
apply_gradients = optimizer.apply_gradients(gvs)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

