#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  # plt ?�於顯示?��?
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
import pandas as pd
from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
import random

early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=1,
                        mode='auto',
                        epsilon=0.0001)

def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data

# 資�?路�?--------------------------------------------------------------------
# base_path = 'C:\\Users\\Irene\\Documents\\ncu\\論�?\\iMFAD'
# cate_dir = base_path + '\\cate_info_squzze'
# data_path = base_path + '\\dataset'
# tarintxt_path = base_path + '\\img_dir.txt'
# tarincate_path_color =  cate_dir+'\\label_color.npy'
# tarincate_path_category =  cate_dir+'\\label_category.npy'
# tarincate_path_style =  cate_dir+'\\label_style.npy'
# tarincate_path_sleeve =  cate_dir+'\\label_sleeve.npy'
# tarincate_path_pattern =  cate_dir+'\\label_pattern.npy'

base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
train_top_dir_file = base_path + '/train/train_top_dir_file.txt'
train_top_label_file = base_path + '/train/train_top_label_file.txt'

train_x_file = base_path + '/train/train_x_file_v2.txt'
train_y_file = base_path + '/train/train_y_file_v2.txt'
val_x_file = base_path + '/validation/val_x_file.txt'
val_y_file = base_path + '/validation/val_y_file.txt'

val_path = base_path + '/validation'
val_label_dir = base_path + '/validation/annos'
val_top_dir_file = base_path + '/validation/val_dir_file.txt'
val_top_label_file = base_path + '/validation/val_label_file.txt'

# 製�?訓練資�? 標籤&資�???-----------------------------------------------------
img_per_amount = 5400#21600

def preprocess(x_path, data_path, x_data_path, y_data_path, name, group_num):
    x_data = ReadFile(x_data_path)
    y_data = ReadFile(y_data_path)
    #y_data = np.load(y_data_path, allow_pickle=True)
    #x_data = x_data[:200]
    #y_data = y_data[:200]
    print("---x_data len = ",name, len(x_data))
    print("---y_data len = ",name, len(y_data))

    # ?�新?��?資�?
    state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(state)
    state = np.random.get_state()
    np.random.shuffle(y_data)
    
    print("---x_data len = ",name, len(x_data))
    print("---y_data len = ",name, len(y_data))

    #設�?input 維度
    dim1 = 128
    dim2 = 128
    val_non_exist = []

    ##---- 設�?資�???---------------------------------------
    # 存�?練�??�x Npy----------------------------------------
    x_after_data = np.zeros((img_per_amount, dim1, dim2, 3))
    k = 0   # 第k筆npy
    file_cot = 0    # ?��?檔�???    
    y_after_data = np.zeros((img_per_amount))
    output_file = open(data_path+'/'+ 'label'+ name +'.txt', 'w')
    x_output_file = open(data_path+'/'+ 'train'+ name +'.txt', 'w')
    non_exist = []
    
    for i in range(len(x_data)):
        if (x_data[i].find('Up') == -1): 
            #print("pass",i)
            pass
        else:    
            #print("not break",i)
            x = x_path+x_data[i]
            if os.path.isfile(x) and x != []:
                    if int(y_data[i])<= 6 :#and int(y_data[i])< 10: #'y_data[i] !='10' and y_data[i] !='11' and y_data[i] !='12' and y_data[i] !='13'
                        if y_data[i]=='6':
                            y_after_data[k] = '5'
                        else:
                            y_after_data[k] = y_data[i]

                    output_file.write(str(y_after_data[k])+'\n')
                    x_output_file.write(str(x)+'\n')
                    img = cv2.imread(x)#讀??                    
                    img = cv2.resize(img, (dim1, dim2), interpolation=cv2.INTER_LINEAR)
                    img = img_to_array(img)
                    x_after_data[k] = img
                    k += 1
                    #else:
                    #    print("--data 10111213")
            else:
                non_exist.append(i)
                print("--Delete Not File--")                    
            
            if k == img_per_amount:
                print("file_count", file_cot+1)
                non_exist = list(set(non_exist))
                print("non_exist",non_exist, len(non_exist))  
                print(f'x training data', x_after_data.shape)
                np.save(os.path.join(data_path,'inputs' + str(file_cot + 1) + name + '.npy'), x_after_data)
                
                print("Before One Hot y_after_data[k-1]",y_after_data[k-1], y_after_data.shape)
                # One Hot Encoding
                y_after_data = np_utils.to_categorical(y_after_data, group_num)
                print("After One Hot y_after_data[k-1]",y_after_data[k-1], y_after_data.shape)
                np.save(os.path.join(data_path,'labels' + str(file_cot + 1) + name + '.npy'), y_after_data)
    
                k = 0
                file_cot += 1
                y_after_data = np.zeros((img_per_amount))
    print(k)
    output_file.close() 
    x_output_file.close() 



# In[72]:

#preprocess(train_path+'/',train_path, train_top_dir_file, train_top_label_file, 'train_v2', 14) 
#preprocess(val_path+'/',val_path, val_top_dir_file, val_top_label_file, 'val_v2', 14) 
#preprocess(train_path+'/image_new/', train_path, train_x_file, train_y_file, 'train_up_human', 7) 
preprocess(val_path+'/image_new/', val_path, val_x_file, val_y_file, 'val_up_human', 7) 
#preprocess(val_path+'/image_new/', val_path, val_x_file, val_y_file, 'val_down', 4) 



