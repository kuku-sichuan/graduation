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
    pos_pred = pred[pos]
    neg = ~pos
    neg_pred = pred[neg]
    pos_loss = -ratio * np.sum(np.log(pos_pred[:,1]))
    neg_loss = -np.sum(np.log(neg_pred[:,0]))
    loss  = (pos_loss + neg_loss) / N
    return loss
def accuracy(pred,y):
    """
    :param pre: the value get from the network!
    :param y: the true labels!
    :return: the pos_acc and the neg_acc
    """
    #Consider about if the samples doesn't have the pos sample!
    pos = y[:,1] > 0
    if (np.sum(pos) == 0):
        pos_acc = 1.0
    else:
        pos_y = y[pos]
        pos_pred = pred[pos]
        pos_acc = np.equal(np.argmax(pos_y, axis=1), np.argmax(pos_pred, axis=1))
    neg = ~pos
    neg_y = y[neg]
    neg_pred = pred[neg]
    neg_acc = np.equal(np.argmax(neg_y,axis=1),np.argmax(neg_pred,axis=1))
    return pos_acc,neg_acc

def output():

