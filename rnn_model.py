import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# weights initial!
def weight_init(shape):
    initial = tf.random_uniform(shape,minval=-np.sqrt(5) * np.sqrt(1.0 / shape[0]),maxval=\
                                np.sqrt(5)*np.sqrt(1.0 / shape[0]))
    return tf.Variable(initial, trainable=True)

# bias initial!
def bias_init(shape):
    initial = tf.constant(0.01,shape)
    return tf.Variable(initial)



