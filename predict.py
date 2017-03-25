import tensorflow as tf
import numpy as np
from deal_data import *

# the files of data!
x_name = '0'
# max depth of drill!
max_T = 102
batch_size = 50
epoch = 100
num_epoch_iters = 7
saver = tf.train.import_meta_graph('model1.ckpt.meta')
graph = tf.get_default_graph()
input = graph.get_tensor_by_name("Input/input:0")
# input = graph.get_operation_by_name('input')
# pred = graph.get_operation_by_name('Forcast')
# set up the sess
with tf.Session(graph=graph) as sess:
    step = 0
    features, labels = load_data(x_name, max_T)
    features, means_t, std_t = preprocess_feature1(features)
    print 'the means of train:'
    print means_t
    print 'the std of train:'
    print std_t
    labels = preprocess_data(labels)
    get_batch = next_batch(features, labels)
    # care that the weight load must be done in the session!
    batch_x, batch_y = get_batch.reset(batch_size, step, num_epoch_iters)
    feedDict = {input: batch_x}
    k = sess.run(graph.get_operations(),feed_dict=feedDict)
    print k
    # features, _ = load_data(x_name,max_T)
    # feedDict = {input: pred_x}
    # pred = sess.run(test_rnn.predict, feed_dict=feedDict)