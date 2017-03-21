# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.ctc import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from utils import load_batched_data


INPUT_PATH = './line_data/image'    # directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './line_data/label/'  # directory of nCharacters 1-D array .npy files

gpu_number = 0        #使用第几块GPU
# ## # Learning Parameters
lr_initial = 0.001    #初始学习率
lr_decay_ratio = 0.9  
decay_step = 100  
decay_num = 12        #衰减次数
momentum = 0.9   
nEpochs = 5000
batchSize = 128
snapshot = 40         #保存模型的频率

# ## # Network Parameters
nFeatures = 1024       #与输入序列向量的长度一致
nHidden = 128
nClasses = 81         #类别数+1.(class number,plus the "blank" for CTC)
num_layers = 4   	  #lstm层数
keep_prob = 1.0   	  #设置dropout的值，dropout = 1-keep_prob

# ## # Load data
#每个样本的Width最好一致，否则loaddata会对样本做填充，耗费时间
print('Loading data')
batchedData, maxTimeSteps, totalN = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)

# ## # Define graph
print('Defining graph')
graph = tf.Graph()

#变量值都存放在cpu中，传输数据到GPU的速度快
with graph.as_default(), tf.device('/cpu:0'):
    # NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow
    learningRate = tf.placeholder(tf.float32)    
    # ## # Graph input
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
    # Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    inputXrs = tf.reshape(inputX, [-1, nFeatures])
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_feature)
    inputList = tf.split(0, maxTimeSteps, inputXrs)
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))
    #seqLengths refers to the cols of each image
    # ## # Weights & biases
    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    biasesClasses = tf.Variable(tf.zeros([nClasses]))

    # ## # Network
    fw_cell = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    bw_cell = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
    bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)

    forwardH1 = rnn_cell.MultiRNNCell([fw_cell] * num_layers, state_is_tuple=True)
    backwardH1 = rnn_cell.MultiRNNCell([bw_cell] * num_layers, state_is_tuple=True)
    fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32,
                                       scope='BDLSTM_H1')

    with tf.device('/gpu:'+str(gpu_number)):
        with tf.name_scope('tower_1') as scope:
            fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
            outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

            logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

            # Optimizing
            logits3d = tf.pack(logits)
            loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
            opt = tf.train.MomentumOptimizer(learningRate, momentum)
            optimizer = opt.minimize(loss)
            grads = opt.compute_gradients(loss)
			
            # 记录grads,loss,lr的值
            for grad,var in grads:
                if grad is not None:
					tf.histogram_summary(var.op.name + '/gradients', grad)
            tf.scalar_summary("loss", loss) 
            tf.scalar_summary("lr", learningRate)
            # ## # Evaluating
            logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
            predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
            errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
                        tf.to_float(tf.size(targetY.values))


# Run session
with tf.Session(graph=graph, config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False))as session:
    print('Initializing')
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs/", session.graph)
    tf.initialize_all_variables().run()

    iter_count = 1  #记录训练次数
    saver = tf.train.Saver()

    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData))   # randomize batch order
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            #调学习率
            if epoch <= decay_num*decay_step: 
                lr = lr_initial*(lr_decay_ratio)**(epoch/decay_step)
            else:
                lr = lr_initial*(lr_decay_ratio)**decay_num

            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, 
                        learningRate:lr}
            _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
            assert not np.isnan(l), 'Model diverged with loss = NaN'
            #visualize
            if iter_count % 5 == 0:
                result = session.run(merged, feed_dict=feedDict)
                writer.add_summary(result, iter_count)
            iter_count += 1

            print(np.unique(lmt))#print unique argmax values of first sample in batch;should be blank for a while
            # then spit out target values
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
                print('lr:',lr,'  epcoch:',epoch)
            batchErrors[batch] = er*len(batchSeqLengths)
        epochErrorRate = batchErrors.sum() / totalN
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)
		#保存模型
        if (epoch%snapshot) == 0:  
            save_path = saver.save(session,'HandWriting_gpu2_'+str(epoch)+'.tfmodel')
