import tensorflow as tf
import numpy as np

def write(pre):
    i = 0
    f=open('pred0.txt','a')
    while(i < 10):
        mmin = tf.reduce_min(pre[i,:])
        print mmin
        f.writelines(str(mmin))
        i +=1
    f.close()
a = np.random.randint(1,5,(10,5))
a = a * 3

with tf.Session() as sess:
    print a
    print sess.run(min(a[0,:]))
    sess.run(write(a))