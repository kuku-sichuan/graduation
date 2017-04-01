# coding=utf-8
import xlrd
import xlwt
import os
import numpy as np
import tensorflow as tf


def ptxt2xl(file_name):

    """
    input:
    file_name: name which include everything that we wanted!
    fuction:
    convert the data from txt to excel!
    """
    # this need to be made by myself!
    f = open(file_name + '/0.txt')
    line = f.readline()
    i = 0
    j = 0
    pre_x= -1
    pre_y= -1
    num_dril = -1
    num_page = 1

    # crate the execl book
    path = file_name + '/' + '%d.xlsx'%(num_page)
    data = xlrd.open_workbook(path)
    book = xlwt.Workbook()
    sheet1 = book.add_sheet('sheet1',cell_overwrite_ok=True)
    while line:
        #get the x_val and clear
        num = line.find(';')
        x_val = line[:num]
        x_val = float(x_val)
        line = line[num+1:]

        #get the y_val and clear
        num = line.find(';')
        y_val = line[:num]
        y_val = float(y_val)
        line = line[num+1:]

        #get the z_val and clear
        z_val = line[:-1]
        z_val = float(z_val)


        if ((y_val == pre_y) and (x_val == pre_x)):
            sheet1.write(i,j,x_val)
            sheet1.write(i,j + 1,y_val)
            sheet1.write(i,j + 2,z_val)
            i += 1
            j = num_dril * 3
        # next drill
        else:
            pre_x = x_val
            pre_y = y_val
            num_dril += 1
            j = num_dril * 3
            if (j < 252):
                #jump the circulation!
                i = 0
                sheet1.write(i,j,x_val)
                sheet1.write(i,j + 1,y_val)
                sheet1.write(i,j + 2,z_val)
                i += 1
            else:
                book.save(path)
                num_page += 1
                path = file_name + '/' + '%d.xlsx' % (num_page)
                data = xlrd.open_workbook(path)
                book = xlwt.Workbook()
                sheet1 = book.add_sheet('sheet1', cell_overwrite_ok=True)
                i = 0
                j = 0
                sheet1.write(i,j,x_val)
                sheet1.write(i,j + 1,y_val)
                sheet1.write(i,j + 2,z_val)
                i += 1
                num_dril = 0
                pre_x = x_val
                pre_y = y_val
        line = f.readline()
    book.save(path)

def pload_data(file_name,max_T):
    """
    input:
    file_name： name of excel file
    max_T:   is the max depth of drill

    output:
    features:the data of drill N x T x D.
    N: number of drill
    T: the depth of a drill
    D: the dimension of the mineral information.

    fuction:
    read the drill data from excel then organize it into
    N x T x D numpy.array data structure.
    """
    mine_data = []
    for j in os.listdir(file_name):
        if j.endswith('xlsx'):
            path = file_name + '/' + j
            print j
        else:
            continue
        try:
            data = xlrd.open_workbook(path)
        except Exception,e:
            print str(e)
        sheet1 = data.sheet_by_index(0)
        for i in xrange(0,sheet1.ncols - 3,3):
          drill_data = []
          x_val = sheet1.col_values(i)
          y_val = sheet1.col_values(i+1)
          z_val = sheet1.col_values(i+2)
          while (x_val[-1] == ''):
              x_val.pop()
              y_val.pop()
              z_val.pop()
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

          # add all the data about a drill into a list
          drill_data.append(x_val)
          drill_data.append(y_val)
          drill_data.append(z_val)

          #convert the list to T x (D+1)
          drill_data = np.vstack(drill_data) # Dx T
          drill_data = drill_data.T  # T x D
          mine_data.append(drill_data)
    # compose the mine_data by drill_data
    features = np.array(mine_data) # N x T x D
    return features

def dense(features,max_T,D):
    x_max = np.max(features[:,:,0])
    x_min = np.min(features[:,:,0])
    y_max = np.max(features[:,:,1])
    y_min = np.min(features[:,:,1])
    x_len = len(np.arange(x_min,x_max,2))
    y_len = len(np.arange(y_min,y_max,2))
    num_dense = x_len * y_len
    dense_feat = np.zeros((num_dense,max_T,D))
    i = 0
    for x in np.arange(x_min,x_max,2):
        for y in np.arange(y_min,y_max,2):

            #get the surface data for interpolation
            surface = features[:,0,:]

            #deal the x_axis
            dist_x = np.abs(surface[:, 0] - x)
            dist_y = np.abs(surface[:, 1] - y)
            dist = dist_x + dist_y
            locat = np.argmin(dist)
            dense_feat[i, :, 0] = x
            dense_feat[i, :, 1] = y
            dense_feat[i, :, 2] = features[locat,:,2]
            i += 1
    return dense_feat




def write2txt(s,batch_size,pred,orig_feature,pred_txt,max_T):
    """

    :param s: is the num of item
    :param batch_size:
    :param pred: the tensor of rnn_network output
    :param orig_feature: tensor of origin features
    :param pred_name: the txt for output
    :param max_T:the drill max depth!
    :return: evey batch has empty drill
    function:
    write the boundary into txt
    """
    num = s * batch_size
    # pred is batch_size * num_depth
    i = 0
    f = open(pred_txt,'a')
    while (i < batch_size):
        # location is each drill !
        each_num = num + i
        j = 0
        while (j < max_T):
            loc = orig_feature[each_num,j,:]
            f.writelines(str(loc[0]) + ', ' + str(loc[1]) + ', ' + str(loc[2]) +', ' + str(pred[i,j]) + '\n')
            j += 1
        i += 1
    f.close()


def write2txt2(output,orig_feature,max_T,pred_txt):
    """
    :param output: all the output for to write!
    :param orig_feature:
    :param max_T:
    :param pred_txt:
    :return:
    """
    N, T = output.shape
    f = open(pred_txt, 'a')
    for n in xrange(N):
        for t in xrange(T):
            f.writelines(str(orig_feature[n,t,0]) + ', ' + str(orig_feature[n,t,1]) + ', '
                         + str(orig_feature[n,t,2]) + ', ' + str(output[n, t]) + '\n')
    f.close()








