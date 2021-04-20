# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import re
import detectColor
import time
from keras.utils import np_utils

#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
save_path_body = base_path + '/train/img_body/' 
save_path_hand = base_path + '/train/img_hand/' 

train_x_file = base_path + '/train_x_0413_hand.txt'
train_y_file = base_path + '/train_y_0413_hand.txt'

train_x_sleeve = open(train_path+'/train_x_0413_sleeve.txt',"w")
train_y_sleeve = open(train_path+'/train_y_0413_sleeve.txt',"w")


def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

def ReadFile_Label(data_path):
    data = []
    label = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line[:-1]
            s = line.split('#')
            data.append(s[0])
            label.append(re.sub('#','',s[1]))
            
    print(len(data),len(label)) 
    return data,label

data,label =ReadFile_Label(base_path + '/train/train_file_clean_0413.txt')

def SearchImginDoc():
    for filename in os.listdir(save_path_hand):
        filename = re.sub('.jpg','',filename) + '.jpg'
        if filename in data:
            train_x_file.write(filename+'\n')
            train_y_file.write(label[data.index(filename)]+'\n')
            #print(data.index(filename))

def RiviseIndex_Sleeve():
    x_data = ReadFile(train_x_file)
    y_data = ReadFile(train_y_file)
    x_after_data = []
    y_after_data = []
    for i in range(len(x_data)):
        if int(y_data[i])<=6 :#and int(y_datga[i])< 10: #'y_data[i] !='10' and y_data[i] !='11' and y_data[i] !='12' and y_data[i] !='13'
            x_after_data.append(x_data[i]) 
            if int(y_data[i])<=2 :
                y_after_data.append(y_data[i]) 
            elif y_data[i]=='3':
                y_after_data.append('1') 
            elif y_data[i]=='4':
                y_after_data.append('2')
            elif y_data[i]=='5' or y_data[i]=='6':
                y_after_data.append('3')
            else:
                y_after_data.append('0')

    for i in range(len(x_after_data)):
        train_x_sleeve.write(x_after_data[i]+'\n')
        train_y_sleeve.write(y_after_data[i]+'\n')

if __name__ == '__main__':
    RiviseIndex_Sleeve() 
# x_data = ReadFile(train_x_file)
# y_data = ReadFile(train_y_file)
# group_num = 4
# dim1 = 128
# dim2 = 128
# x_path = train_path + '/img_hand/'
# x_after_data = np.zeros((len(x_data), dim1, dim2, 3))
# y_after_data = np.zeros((len(x_data)))
# # x_data = ReadFile(train_x_s)
# # y_data = ReadFile(train_y_s)
# # v_x_data = ReadFile(val_x_s)
# # v_y_data = ReadFile(val_y_s)
# for k in range(len(x_data)): 
#     #print("not break",i)
#     x = x_path+x_data[k]
#     if os.path.isfile(x) and x != []:
#             img = cv2.imread(x)#讀??                    
#             img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
#             img = img_to_array(img)
#             x_after_data[k] = img
#     print("Before One Hot y_data[k-1]",y_data[k])
#     # One Hot Encoding
#     y_after_data = np_utils.to_categorical(y_data[k], group_num)
#     print("After One Hot y_after_data[k-1]",y_after_data[k], y_after_data.shape)
# np.save(os.path.join(data_path,'train_0413_sleeve_inputs.npy'), x_after_data)
# np.save(os.path.join(train_path,'train_0413_sleeve_labels.npy'), y_after_data)
