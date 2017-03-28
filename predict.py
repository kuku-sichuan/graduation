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

def write2txt(s,batch_size,pred,orig_feature,pred_txt):
    """

    :param s: is the num of item
    :param batch_size:
    :param pred: the tensor of rnn_network output
    :param orig_feature: tensor of origin features
    :param pred_name: the txt for output
    :return: evey batch has empty drill
    function:
    write the boundary into txt
    """
    num = s * batch_size
    # pred is batch_size * num_depth
    con_group = tf.ones_like(pred)
    ismine = tf.equal(pred,con_group)
    k = 0
    empty = 0
    f = open(pred_txt,'a')
    while (k < batch_size):
        # location is each drill !
        location = tf.where(ismine[k,:])
        if (location.shape[0] != 0):
            # this may be need to change!
            # for complex condition,this is not work!
            min_loc = tf.reduce_min(location)
            max_loc = tf.reduce_max(location)
            each_num = num + k
            min_loc = orig_feature[each_num,min_loc,:]
            max_loc = orig_feature[each_num,max_loc,:]
            f.writelines(str(min_loc[0]) + ',' + str(min_loc[1]) + ',' + str(min_loc[2]) +';' +
                         str(max_loc[0]) + ',' + str(max_loc[1]) + ',' + str(max_loc[2]) + '\n')
            k += 1
        else:
            # if location is empty how to deal!
            empty += 1
    f.close()
    return empty










