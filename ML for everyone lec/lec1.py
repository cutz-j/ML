import tensorflow as tf

hello = tf.constant("hello, tensorFlow!")
sess=tf.Session()
#print(sess.run(hello))

node1=tf.constant(3.0, tf.float32)
node2=tf.constant(4.0)
node3=tf.add(node1, node2)

#print("node1:", node1)

sess=tf.Session()
#print(sess.run([node3]))

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

adder_node=a+b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[3,4], b:[2,5]}))