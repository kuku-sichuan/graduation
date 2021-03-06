from __future__ import print_function
import tensorflow as tf
import numpy as np
from deal_data import *
from rnn_model import RNN_lstm
from predict import *

# set the param
############################################
# the files of data!
x_name = '0'
# max depth of drill!
max_T = 102
file_name = 'pred0'
pred_txt = "pred0.txt"

# Network Paramters
n_input = 3 # D
n_steps = 102 # the num of inputs
n_hidden1 = 400 # the num of time sequence! maybe this need to change!
n_hidden2 = 100
n_stacks = 2
n_class = 2

start_learning_rate = 0.001
# we set the decay of learning rate!\
# trianing part!
batch_size = 50
epoch = 150
num_epoch_iters = 7
display_step = 7
train_iters = epoch * num_epoch_iters
train = True
nn_type = 'rnn'
############################################
# set up the graph

graph = tf.Graph()
with graph.as_default():
    with tf.device('/gpu:0'):
      test_rnn = RNN_lstm(n_input,n_steps,n_hidden1,n_hidden2,n_stacks,n_class,start_learning_rate,batch_size,nn_type,train)
      init = tf.global_variables_initializer()
      saver = tf.train.Saver()
##############################################
# set up the sess
with tf.Session(graph=graph, config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
    features, labels = load_data(x_name, max_T)
    features, means_t, std_t = preprocess_feature1(features)
    if train:
        merge = tf.summary.merge_all()

        train_write = tf.summary.FileWriter('checkpoint/train',sess.graph)
        test_write = tf.summary.FileWriter('checkpoint/test')

        sess.run(init)
        #########################################
        # pre data deal!
        print ('the means of train:')
        print (means_t)
        print ('the std of train:')
        print (std_t)
        labels = preprocess_data(labels)
        get_batch = next_batch(features,labels)

        step = 0
        # Keep training until reach max iterations
        test_x, test_y = get_batch.sample_test(0.05)
        while step < train_iters:
            # second to optim
            batch_x, batch_y = get_batch.reset(batch_size,step,num_epoch_iters)
            feedDict = {test_rnn.input:batch_x, test_rnn.labels:batch_y}
            sess.run(test_rnn.optim,feed_dict= feedDict)
            result = sess.run(merge, feed_dict=feedDict)

            train_write.add_summary(result, step)
            if step % display_step == 0:
                n,p = sess.run([test_rnn.neg_acc,test_rnn.pos_acc], feed_dict=feedDict)
                loss = sess.run(test_rnn.loss,feed_dict=feedDict)
                print('Iter' + str(step) + 'Minibatch loss' + \
                      '{:.6f}'.format(loss) + ',Training pos_Accuracy' + \
                      '{:.5f}'.format(p) + ',Training neg_Accuracy' + \
                      '{:.5f}'.format(n))
                testDict = {test_rnn.input:test_x, test_rnn.labels:test_y}
                tn,tp,test_summary = sess.run([test_rnn.neg_acc,test_rnn.pos_acc,merge], feed_dict=testDict)

                test_write.add_summary(test_summary,step)
                print ('Testing pos_Accuracy' + \
                    '{:.5f}'.format(tp) + ',Testing neg_Accuracy' + \
                    '{:.5f}'.format(tn) + '\n')
            step += 1
        save_path = saver.save(sess,'checkpoint/model1.ckpt')
        print ('Optimization finished!')
    else:
        output = np.zeros(())
        ckpt = tf.train.get_checkpoint_state('checkpoint/train')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            # pre data deal!
            orig_features = pload_data(file_name,max_T)
            orig_features = dense(orig_features,max_T,3)
            output = np.zeros(())
            print (orig_features.shape)
            print (orig_features[0][:5])

            #features which is N * T * D
            now_features =  preprocess_feature2(orig_features,means_t,std_t)
            N,T,D = now_features.shape
            output = np.zeros((N,T))
            print (N,T,D)
            total = np.ceil(N*1.0 / batch_size)
            s = 0

            while(s < total-1):
                print (s)
                batchX = now_features[s*batch_size:(s+1)*batch_size]
                feedDict = {test_rnn.input:batchX}
                pred = sess.run(test_rnn.predict,feed_dict=feedDict)
                output[s*batch_size:(s+1)*batch_size,:] = pred
                # write2txt(s,batch_size,pred,orig_features,pred_txt,max_T)
                s += 1
            batchX = now_features[s*batch_size:]

            #for the reset data how to do!
            n,_,_ = batchX.shape
            time = np.int(np.ceil(N / n))
            order = []
            for i in xrange(time):
                for j in xrange(n):
                    order.append(j)

             # in the txt need people to deal with!
            order = np.array(order[:50])
            batchX = batchX[order]
            feedDict = {test_rnn.input: batchX}
            pred = sess.run(test_rnn.predict, feed_dict=feedDict)
            output[s*batch_size:] = pred[0:n,:]
            #write2txt(s, n, pred, orig_features, pred_txt,max_T)
            write2txt2(output,orig_features,max_T,pred_txt)

        else:
            print ("nothing")
