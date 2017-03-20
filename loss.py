import tensorflow as tf
import numpy as np
def loss(pred, y,ratio):
    """
    :param pred: the noework get the value of the class.shape (N*T) x C
    :param y: the really value of sample (N*T) * C
    :param ratio: the ratio of neg sample to pos sample
    :return: the loss
    """
    pos = y[:,1] > 0
    pos = tf.to_float(pos)

    pos = tf.reshape(pos,(-1,1))
    pos_pred = pred * pos
    temp_p =  pos_pred[:,1]
    pos_loss = -ratio * tf.reduce_sum(tf.log(temp_p))
    neg = y[:,0] > 0
    neg = tf.to_float(neg)

    neg = tf.reshape(neg,(-1,1))
    neg_pred = pred * neg
    neg_loss = -tf.reduce_sum(tf.log(neg_pred[:,0]))
    # care about this need to change with the batch size!
    loss  = (pos_loss + neg_loss) / 50
    return loss
def accuracy(pred,y):
    """
    :param pre: the value get from the network! (N*T) * C
    :param y: the true labels!(N*T) *C
    :return: the pos_acc and the neg_acc
    """
    #Consider about if the samples doesn't have the pos sample!
    pos = y[:,1] > 0
    pos = tf.to_float(pos)

    if (tf.reduce_sum(pos) == 0):
        pos_acc = 1.0
    else:
        max_index =  tf.argmax(pred,axis=1)
        max_index = tf.to_float(max_index)

        pos_pred = pos * max_index
        pos_acc = tf.reduce_sum(pos_pred) / tf.reduce_sum(pos)

    neg = y[:,0] > 0
    neg = tf.to_float(neg)

    min_index = tf.argmin(pred, axis=1)
    min_index = tf.to_float(min_index)
    neg_pred = neg * min_index
    neg_acc = tf.reduce_sum(neg_pred) / tf.reduce_sum(neg)
    return pos_acc,neg_acc
