import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class RNN_lstm(object):
    def __init__(self,input_dims,time_steps,num_hiddens,num_stack,num_class,learning_rate,batch_size):
        self.input_dims = input_dims
        self.time_steps = time_steps
        self.num_hiddens = num_hiddens
        self.num_stack = num_stack
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        with tf.name_scope('Input'):
            self.input = tf.placeholder('float', [None, n_steps, n_input], name='input')
        with tf.name_scope('Label'):
            self.labels = tf.placeholder('float', [None, n_class], name='output')

    def weight_init(self,shape):
        initial = tf.truncated_normal(shape,stddev= 0.1)
        return tf.Variable(initial)
    def bias_init(self,shape):

        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self,var,name):
        with tf.name_scope(name + "_summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/'+ name, mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram('hist/' + name,var)
    def rnn_network(self):
        with tf.name_scope('rnn'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.num_hiddens, forget_bias=1.0, state_is_tuple=True)

            # define first hidden layer!
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_stack, state_is_tuple=True)
            initial_state = cell.zero_state(self.batch_size, tf.float32)  # this mean the batch size
            # outputs: n_steps * batch_size * hidden
            # each time step has a output and state
            outputs, last_states = tf.contrib.rnn.static_rnn(cell, self.input, initial_state)
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            tf.summary.histogram('rnn/' + 'hid_val', outputs)
            return outputs
    def build_net(self):
        outputs = self.rnn_network()
        out = tf.reshape(outputs, [-1, self.num_hiddens])
        with tf.name_scope('Output'):
            weights = self.weight_init([self.num_hiddens,self.num_class])
            biases = self.bias_init([self.num_class])
            pre = tf.matmul(out, weights) + biases  # size is (N * T) * 2
        return pre

