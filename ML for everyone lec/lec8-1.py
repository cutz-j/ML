import tensorflow as tf
import numpy as np
import pprint

''' pprint / array '''
tf.set_random_seed(777)
pp=pprint.PrettyPrinter(indent=4)
#sess=tf.InteractiveSession()

t = np.array([0., 1., 2., 3., 4., 5., 6.])
#pp.pprint(t)

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
#print(tf.shape(t).eval())

''' matmul(행렬곱) vs multiply(브로드캐스팅) '''
matrix1=tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.],[2.]])
#print(tf.matmul(matrix1, matrix2).eval())

''' Reduce mean '''
print(tf.reduce_mean([1.,2.], axis=0).eval())

''' Argmax '''

''' reshape '''

t = np.array([[[0, 1, 2], 
               [3, 4, 5]],
              
              [[6, 7, 8], 
               [9, 10, 11]]])

#print(tf.reshape(t, shape=[-1, 3]).eval())

# squeez
print(tf.squeeze([[1],[2],[3]]).eval())

''' one hot '''
t=tf.one_hot([[0],[1],[2],[0]], depth=3).eval()
print(tf.reshape(t, shape=[-1,3]).eval())

''' casting '''
print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())
print(tf.cast([True, False, 1==1, 0==1], tf.int32).eval())

''' stack '''
x=[1,4]
y=[2,5]
z=[3,6]

print(tf.stack([x,y,z], axis=1).eval())

''' zip '''
for x, y in zip([1,2,3],[4,5,6]):
    print(x,y)









