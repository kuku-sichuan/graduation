import tensorflow as tf
import numpy as np

x = np.array([[0,1],[1,0]])
y = np.array([[1,0],[1,0]])
loss = tf.nn.softmax_cross_entropy_with_logits(y,x)
opt = tf.gradients(loss