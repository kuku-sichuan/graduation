import tensorflow as tf
import numpy as np
def loss(pred, y,ratio):
    """
    :param pred: the noework get the value of the class.shape (N*T) x C
    :param y: the really value of sample (N*T) * C
    :param ratio: the ratio of neg sample to pos sample
    :return: the loss
    """
    N,_ = y.shape
    pos = y[:,1] > 0
    pos = pos.reshape((-1,1))
    pos_pred = pred *pos
    neg = y[:,0] > 0
    neg = neg.reshape((-1,1))
    neg_pred = pred * neg
    pos_loss = -ratio * np.sum(np.log(pos_pred[:,1]))
    neg_loss = -np.sum(np.log(neg_pred[:,0]))
    loss  = (pos_loss + neg_loss) / N
    return loss
def accuracy(pred,y):
    """
    :param pre: the value get from the network! (N*T) * C
    :param y: the true labels!(N*T) *C
    :return: the pos_acc and the neg_acc
    """
    #Consider about if the samples doesn't have the pos sample!
    pos = y[:,1] > 0
    if (np.sum(pos) == 0):
        pos_acc = 1.0
    else:
        pos_pred = pos * np.argmax(pred,axis=1)
        pos_acc = np.sum(pos) / np.sum(pos_pred)
    neg = y[:,0] > 0
    temp_y = tf.transpose
    np.argmax(pred,axis=1)
    neg_acc = np.equal(np.argmax(neg_pred,axis=1),0)
    return pos_acc,neg_acc
