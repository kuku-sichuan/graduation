# coding=utf-8
import xlrd
import xlwt
import numpy as np
import os
import random

def load_data(x_name,max_T):
    """
    input:
    x_name： name of excel file
    max_T:   is the max depth of drill

    output:
    features:the data of drill N x T x D.
    N: number of drill
    T: the depth of a drill
    D: the dimension of the mineral information.
    label : number of drill N x T

    fuction:
    read the drill data from excel then organize it into
    N x T x D numpy.array data structure.
    """
    mine_data = []
    for j in os.listdir(x_name):
        path = x_name + '/' + j
        print j
        try:
            data = xlrd.open_workbook(path)
        except Exception,e:
            print str(e)
        sheet1 = data.sheet_by_index(0)
        for i in xrange(0,sheet1.ncols - 4,4):
          drill_data = []
          x_val = sheet1.col_values(i)
          y_val = sheet1.col_values(i+1)
          z_val = sheet1.col_values(i+2)
          label = sheet1.col_values(i+3)
          while (x_val[-1] == ''):
              x_val.pop()
              y_val.pop()
              z_val.pop()
              label.pop()
          # polish the data into a well-structed data!
          pre_x = x_val[0]
          pre_y = y_val[0]
          pre_z = z_val[-1]
          diff = z_val[-1] - z_val[-2]
          if (len(x_val) != max_T):
              remain = max_T - len(x_val)
              for i in xrange(remain):
                  x_val.append(pre_x)
                  y_val.append(pre_y)
                  z_val.append(pre_z + i * diff)
                  label.append(-1)

          # add all the data about a drill into a list
          drill_data.append(x_val)
          drill_data.append(y_val)
          drill_data.append(z_val)
          drill_data.append(label)

          #convert the list to T x (D+1)
          drill_data = np.vstack(drill_data) # (D+1) x T
          drill_data = drill_data.T  # T x (D + 1)
          mine_data.append(drill_data)

    # compose the mine_data by drill_data
    mine_data = np.array(mine_data) # N x T x (D + 1)
    features = mine_data[:,:,:-1]
    label = mine_data[:,:,-1]
    return features, label

def calss(x_name,max_T):
    features,labels = load_data(x_name,max_T)
    labels = preprocess_data(labels)
    N,T,D = features.shape
    features = features.reshape((N*T, D))
    labels = labels.reshape((-1,2))
    pos = labels[:,1] > 0
    pos_f = features[pos]
    pos_l = labels[pos]
    neg = ~pos
    neg_f = features[neg]
    neg_l = labels[neg]
    return [pos_f,pos_l,neg_f,neg_l]


def txt2xl(t_name,x_name):

    """
    input:
    t_name: txt.name which we need to convert.
    x_name: excel.name which we save the data.
    fuction:
    convert the data from txt to excel!
    """

    f = open(t_name)
    line = f.readline()
    i = 0
    j = 0
    pre_x= -1
    pre_y= -1
    num_dril = -1
    num_page = 1

    # crate the execl book
    path = x_name + '/' + '%d.xlsx'%(num_page)
    data = xlrd.open_workbook(path)
    book = xlwt.Workbook()
    sheet1 = book.add_sheet('sheet1',cell_overwrite_ok=True)
    while line:
        if (j < 252):
            #clear the front invalid information
            num = line.find(':')
            line = line[num+2:]

            #get the x_val and clear
            num = line.find(' ')
            x_val = line[:num]
            x_val = float(x_val)
            line = line[num+2:]

            #get the y_val and clear
            num = line.find(' ')
            y_val = line[:num]
            y_val = float(y_val)
            line = line[num+2:]

            #get the z_val and clear
            num = line.find(' ')
            z_val = line[:num]
            z_val = float(z_val)
            line = line[num+2:]

            #get the label and clear
            label = line[:-1]
            label = float(label)

            if ((y_val == pre_y) and (x_val == pre_x)):
                sheet1.write(i,j,x_val)
                sheet1.write(i,j + 1,y_val)
                sheet1.write(i,j + 2,z_val)
                sheet1.write(i,j + 3,label)
                i += 1
                j = num_dril * 4
            # next drill
            else:
                pre_x = x_val
                pre_y = y_val
                num_dril += 1
                j = num_dril * 4
                #jump the circulation!
                i = 0
                sheet1.write(i,j,x_val)
                sheet1.write(i,j + 1,y_val)
                sheet1.write(i,j + 2,z_val)
                sheet1.write(i,j + 3,label)
                i += 1
            line = f.readline()
        else:
            j = num_dril * 4
            sheet1.write(i - 1, j, )
            sheet1.write(i - 1, j+1, ' ')
            sheet1.write(i - 1, j+2, ' ')
            sheet1.write(i - 1, j+3, ' ')
            book.save(path)
            num_page += 1
            path = x_name + '/' + '%d.xlsx' % (num_page)
            data = xlrd.open_workbook(path)
            book = xlwt.Workbook()
            sheet1 = book.add_sheet('sheet1', cell_overwrite_ok=True)
            num_dril = -1
            pre_x = -1
            pre_y = -1
            j = 0
            i = 0
    book.save(path)
            
def preprocess_feature1(features):
    N, T, D = features.shape
    features = features.reshape(N * T, D)
    means_f = np.mean(features,axis = 0)
    std_f = np.std(features, axis = 0)
    features = (features - means_f) / std_f
    features = features.reshape(N, T, D)
    return features,means_f,std_f

def preprocess_feature2(features,means,std):
    N, T, D = features.shape
    features = features.reshape(N * T, D)
    features = (features - means) / std
    features = features.reshape(N, T, D)
    return features

def preprocess_data(label):
    trans_matrix = np.array([[-1, 6]]) # want to change the gradient ratio between the difference,just change there!
    N, T = label.shape
    label = label.reshape(N,T,1)
    label = np.dot(label, trans_matrix)
    label[label < 0] = 0
    return label

class next_batch:
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels
    def reset(self,batch_size,steps,num_epoch_iters):
        N,T,D = self.features.shape
        s = steps % (num_epoch_iters - 1)
        if (s == 0):
          order = np.arange(N)
          random.shuffle(order)
          self.features = self.features[order]
          self.labels = self.labels[order]
        X = self.features[s*batch_size:(s+1)*batch_size]
        y = self.labels[s*batch_size:(s+1)*batch_size]
        y = y.reshape((-1,2))
        return X,y
    def sample_test(self,ratio):
        """

        :param ratio: the ratio sample for test
        :return: the sample's features and labels!
        """

        N,T,D = self.features.shape
        order = np.arange(N)
        random.shuffle(order)
        self.features = self.features[order]
        self.labels = self.labels[order]
        num_test = np.int(N * ratio)
        test_feat = self.features[:num_test]
        test_label = self.labels[:num_test]
        self.features = self.features[num_test:]
        self.labels = self.labels[num_test:]
        # if the num of test didn't has the batch_size !
        full = np.random.randint(0,num_test,50)
        test_feat = test_feat[full]
        print test_feat.shape
        test_label = test_label[full]
        print test_label.shape
        test_label = test_label.reshape((-1,2))
        return test_feat,test_label

def get_ratio(labels):
    """
    labels: the drill's labels
    function:get the ratio of neg/pos
    """
    N,T = labels.shape
    pos = np.sum(labels[labels == 1])
    print N,T,pos
    ratio = (N*T - pos) / pos
    return ratio


