import tensorflow as tf

a = tf.constant(3)
b = tf.constant(4)

c = tf.add(a,b,name='a_plus_b')
d = tf.multiply(a,b,name='a_mul_b')
e = tf.multiply(c,d,name='c_mul_d')

sess=tf.Session()
summary_writer = tf.summary.FileWriter('log_simple', sess.graph)
sess.run(e)

