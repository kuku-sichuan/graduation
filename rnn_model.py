import tensorflow as tf
import numpy as np

class RNN_lstm(object):
    def __init__(self,input_dims,time_steps,num_hiddens,num_stack,num_class,learning_rate,batch_size,nn_type):
        self.input_dims = input_dims
        self.time_steps = time_steps
        self.num_hiddens = num_hiddens
        self.num_stack = num_stack
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # have two type rnn brnn!
        self.nn_type = nn_type
        with tf.name_scope('Input'):
            self.input = tf.placeholder('float', [None, self.time_steps, self.input_dims], name='input')
        with tf.name_scope('Label'):
            self.labels = tf.placeholder('float', [None, self.num_class], name='output')
        self.build()


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
    def brnn_network(self):
        with tf.name_scope('b_rnn'):
            x = tf.transpose(self.input, [1, 0, 2])  # D*N*T
            # Reshaping to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, self.input_dims])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.split(x, self.time_steps, 0)

            # Forward direction cell
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hiddens, forget_bias=1.0,state_is_tuple=True)
            # Backward direction cell
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hiddens, forget_bias=1.0,state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell] * self.num_stack, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell] * self.num_stack, state_is_tuple=True)
            initial_f = lstm_fw_cell.zero_state(self.batch_size, tf.float32)
            initial_b = lstm_bw_cell.zero_state(self.batch_size, tf.float32)
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,initial_f,initial_b)
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            tf.summary.histogram('brnn/' + 'hid_val', outputs)
            return outputs

    def rnn_network(self):
        with tf.name_scope('rnn'):
            #self.input's shape is
            x = tf.transpose(self.input, [1, 0, 2])  # T*N*D
            # Reshaping to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, self.input_dims])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.split(x, self.time_steps, 0)

            cell = tf.contrib.rnn.BasicLSTMCell(self.num_hiddens, forget_bias=1.0, state_is_tuple=True)
            # define first hidden layer!
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_stack, state_is_tuple=True)
            initial_state = cell.zero_state(self.batch_size, tf.float32)  # this mean the batch size
            # outputs: n_steps * batch_size * hidden
            # each time step has a output and state
            outputs, last_states = tf.contrib.rnn.static_rnn(cell, x, initial_state)
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            tf.summary.histogram('rnn/' + 'hid_val', outputs)
            return outputs
    def train_acc(self,pre):
        with tf.name_scope('accuracy'):
            pos_total = tf.reduce_sum(tf.argmax(self.labels, 1))
            pos_t = tf.to_float(pos_total)
            self.pos_acc = tf.to_float(tf.reduce_sum(tf.argmax(pre, 1) * tf.argmax(self.labels, 1))) / pos_t

            neg_total = tf.reduce_sum(tf.argmin(self.labels, 1))
            neg_t = tf.to_float(neg_total)
            self.neg_acc = tf.to_float(tf.reduce_sum(tf.argmin(pre, 1) * tf.argmin(self.labels, 1))) / neg_t

            tf.summary.scalar('pos_acc', self.pos_acc)
            tf.summary.scalar('neg_acc', self.neg_acc)

    def build(self):
        # build the different network!
        if self.nn_type == 'rnn':
            outputs = self.rnn_network()
        elif self.nn_type == 'brnn':
            outputs = self.brnn_network()
        else:
            # run there exit
            print "the input network didn't have!"
            exit()
        with tf.name_scope('mul_out'):
            if self.nn_type == 'rnn':
                with tf.name_scope('weights'):
                   out = tf.reshape(outputs, [-1, self.num_hiddens])
                   weights = self.weight_init([self.num_hiddens,self.num_class])
                   self.variable_summaries(weights, 'w')
                with tf.name_scope('biases'):
                   biases = self.bias_init([self.num_class])
                   self.variable_summaries(biases,'b')
                pre = tf.matmul(out, weights) + biases  # size is (N * T) * 2
            elif self.nn_type == 'brnn':
                with tf.name_scope('weights'):
                    out = tf.reshape(outputs, [-1, 2 * self.num_hiddens])
                    weights = self.weight_init([2 * self.num_hiddens, self.num_class])
                    self.variable_summaries(weights, 'w')
                with tf.name_scope('biases'):
                    biases = self.bias_init([self.num_class])
                    self.variable_summaries(biases, 'b')
                pre = tf.matmul(out, weights) + biases  # size is (N * T) * 2
        with tf.name_scope('Softmax'):
            pre = tf.nn.softmax(pre)
        with tf.name_scope('loss'):

            # trans there need to be more care about the change!
            #trans = np.array([[1, 0], [0, 1]]).astype('float32')
            #temp_y = tf.matmul(self.labels, trans)

            temp_y = self.labels
            self.loss = tf.reduce_mean(-tf.reduce_sum(temp_y*tf.log(pre),reduction_indices=[1]), name='Loss')
            tf.summary.scalar('loss', self.loss)
        with tf.name_scope('predict'):
            pred = tf.reshape(pre, (self.batch_size,self.time_steps,-1))
            self.predict = tf.argmax(pred,2)

        with tf.name_scope('trian'):
            self.train_acc(pre)
            with tf.name_scope('learn_rate'):
                global_step = tf.Variable(0,trainable=False)
                self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                                100, 0.96, staircase=True)
                tf.summary.scalar('learning_rate', self.learning_rate)
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optim = opt.minimize(self.loss,global_step=global_step)
        with tf.name_scope('grad'):
            self.grads = opt.compute_gradients(self.loss)
            for grad, var in self.grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)




