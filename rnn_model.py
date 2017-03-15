import tensorflow as tf
import functools

# for the code be more simple and reasonable! we build the decorator!

import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator



class model:

    def __init__(self,features,labels = None,hid_layers):
        """
        :param hid_layers:the number of hidden layers
        :param features: is the drill data which has deal,(N,T,D)
        :param labels: (N,T,C)
        """
        self.features = features
        self.labels = labels
        self.hid_layers = hid_layers
        self.prediction
        self.loss
    @lazy_property
    def predict(self):
        N, T, D = self.features.shape
        # permuting batch_size and n_steps
        x = tf.transpose(self.features, [1, 0, 2])
        #reshape to(T*N,D)
        x = tf.reshape(x, [-1, D])
        #split to get a list of T tensors of shape(N,D)
        x = tf.split(x, T, 0)

        #define a lstm cell
        lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(T, forget_bias=1.0,state_is_tuple=True)

        #define first hidden layer!
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1] * self.hid_layers)
        self._initial_state = cell.zero_state(N,tf.float32)



