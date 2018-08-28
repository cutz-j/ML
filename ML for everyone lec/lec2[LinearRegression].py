import tensorflow as tf

#X=[1,2,3]
#Y=[1,2,3]

# 매개변수
X=tf.placeholder(tf.float32, shape=[None])
Y=tf.placeholder(tf.float32, shape=[None])

# tf변수
W=tf.Variable(tf.random_normal([1]), name="weight")
b=tf.Variable(tf.random_normal([1]), name="bias")

hypothesis=X*W+b

# 오차 최소제곱
cost=tf.reduce_mean(tf.square(hypothesis-Y))

# train
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                 feed_dict={X:[1,2,3,4,5], Y: [2.1,3.1,4.1,5.1,6.1]})
    if step % 20==0:
        print(step, W_val, b_val)
        
# Test
print(sess.run(hypothesis, feed_dict={X:[5]}))
        
#for step in range(2001):
#    sess.run