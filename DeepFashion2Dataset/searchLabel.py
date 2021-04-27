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

train_x_file_dir = train_path + '/0420_train_x.txt'
train_y_file_dir = train_path + '/0420_train_y.txt'

train_x_sleeve = open(train_path+'/0420_train_x_sleeve.txt',"w")
train_y_sleeve = open(train_path+'/0420_train_y_sleeve.txt',"w")


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

def SearchImginDoc(data, label,train_x_file,train_y_file):
    for filename in os.listdir(save_path_hand):
        #filename = re.sub('.jpg','',filename) + '.jpg'
        #print(filename)
        if filename in data:
            train_x_file.write(str(filename)+'\n')
            train_y_file.write(label[data.index(str(filename))]+'\n')
            #print(data.index(filename))

def RiviseIndex_Sleeve():
    x_data = ReadFile(train_x_file_dir)
    y_data = ReadFile(train_y_file_dir)
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
    # data,label =ReadFile_Label(base_path + '/train/train_file_clean_0420.txt') 
    # train_x_file = open(train_x_file_dir,"w")
    # train_y_file = open(train_y_file_dir,"w")
    # SearchImginDoc(data,label,train_x_file,train_y_file)

    RiviseIndex_Sleeve()