from deal_data import load_data,get_ratio,preprocess_data
import xlrd,xlwt
import os
import numpy as np

x_name = '0'
t_name = '0.txt'
features, labels = load_data(x_name,max_T=102)
a  = preprocess_data(labels)
print a.shape
ratio = get_ratio(labels)
print ratio



