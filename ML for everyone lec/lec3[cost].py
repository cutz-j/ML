import tesnorflow as tf
import matplotlib.pyplot as plt

X=[1,2,3]
Y=[1,2,3]

W = tf.placeholder(tf.float32)
hypothesis=W*X

cost=tf.reduce.mean(tf.square(hypothesis-W))

sess=tf.Session()

sess.run(tf.global_variables_initializer())

W_val=[]
cost_val=[]
for i in range(-30, 50):
    feed_W=i*0.1
    cuur_cost, curr_W=sess.run[]