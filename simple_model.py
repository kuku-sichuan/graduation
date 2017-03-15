import tensorflow as tf
from tensorflow.python.ops import rnn
from deal_data import *
import numpy as np

# the files of data!
x_name = '0'
# max depth of drill!
max_T = 102

# Network Paramters
n_input = 3 # D
n_steps = 102 # the num of inputs
n_hidden = 102 # the num of time sequence! maybe this need to change!
n_stacks = 2
n_class = 2

#tf Graph input
x = tf.placeholder('float',[None, n_steps, n_input])
y = tf.placeholder('float',[None, n_class])

#define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_class]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_class]))
}

def RNN(x,weights, biases,n_hidden,max_T):

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [2, 0, 1]) # D*N*T
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1,n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    # define first hidden layer!
    cell = tf.contrib.rnn.MultiRNNCell([cell] * n_stacks,state_is_tuple=True)
    initial_state = cell.zero_state(50, tf.float32) # this mean the batch size
    # outputs: n_steps * batch_size * hidden
    # each time step has a output and state
    outputs, last_states = tf.contrib.rnn.static_rnn(cell, x, initial_state)
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2]) # convert the tensor into N*T*H

    out = tf.reshape(outputs, [-1, n_hidden])
    pre = tf.matmul(out,weights['out']) + biases['out'] # size is (N * T) * 2
    return pre

pred = RNN(x, weights,biases,n_hidden,max_T)

start_learning_rate = 0.005
# we set the decay of learning rate!\
# trianing part!
batch_size = 50
epoch = 100
num_epoch_iters = 7
display_step = 7
train_iters = epoch * num_epoch_iters

# make the learning to decay!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(start_learning_rate,global_step,20,0.9)

# Define loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# store the model
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    features,labels = load_data(x_name,max_T)
    features,means_t,std_t = preprocess_feature1(features)
    print 'the means of train:'
    print means_t
    print 'the std of train:'
    print std_t
    labels = preprocess_data(labels)
    get_batch = next_batch(features,labels)
    # Keep training until reach max iterations
    while step < train_iters:
        batch_x, batch_y = get_batch.reset(batch_size,step,num_epoch_iters)
        sess.run(optimizer,feed_dict={x: batch_x,y:batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x, y:batch_y})
            print 'Iter' + str(step) + 'Minibatch loss' + \
                '{:.6f}'.format(loss) + ',Training Accuracy' + \
                '{:.5f}'.format(acc)
        step += 1
    save_path = saver.save(sess,'0model')
    print 'Optimization finished!'
