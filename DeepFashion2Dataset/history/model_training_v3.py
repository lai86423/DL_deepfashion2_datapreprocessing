#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  # plt 用於顯示圖片
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
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=1,
                        mode='auto',
                        epsilon=0.0001)

# 資料路徑--------------------------------------------------------------------
#base_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\The iMaterialist Fashion Attribute Dataset'
#data_path = 'C:\\Users\\Irene\\Documents\\ncu\\論文\\The iMaterialist Fashion Attribute Dataset\\dataset_v0119'
#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
val_path = base_path + '/validation'

# Model -----------------------------------------------------------------------
model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#freeze some layers
for layer in model_resnet.layers[:-12]:
     # 6 - 12 - 18 have been tried. 12 is the best.
     layer.trainable = False
#model_resnet.trainable = False

#build the category classification branch in the model
x = model_resnet.output
x = layers.Dropout(0.5)(x)
#cate
x1 = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
y1 = Dense(14, activation='softmax', name='category')(x1)


#create final model by specifying the input and outputs for the branches
final_model = Model(inputs=model_resnet.input, outputs=[y1, y2, y3])

#print(final_model.summary())

#opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
opt = Adam(learning_rate=0.01)

final_model.compile(optimizer=opt,loss={'category':'categorical_crossentropy'
                                        },
                    metrics={'category':['accuracy','top_k_categorical_accuracy'] #top_k_categorical_accuracy
                            }
                    ) #default:top-5

# color = []
# category = []
# style = []
# style = []
# sleeve = []
# pattern = []
#批次訓練函數
def generate_arrays_from_file(trainpath,set_len,file_nums,has_remainder=0,batch_size=32):
    
    '''
    :param trainsetpath: 訓練集路徑
    :param set_len: 訓練文件的圖片數量
    :param file_nums: 訓練文件數量
    :param has_remainder: 是否有餘數，即has_remainder = 0 if set_len % batch_size == 0 else 1
    :param batch_size: 批次大小
    :return: 
    '''
    cnt = 0 
    pos = 0
    inputs = None
    #labels = None
    labels_color = None
    labels_style = None
    labels_category = None
    while 1:
        if cnt % (set_len // batch_size+has_remainder) == 0:  #判斷是否讀完一整個文件
            pos = 0
            seq = cnt // (set_len // batch_size + has_remainder) % file_nums #此次讀取第seq個文件
            del inputs, labels_color, labels_style, labels_category
            inputs = np.load(os.path.join(trainpath, 'inputs' + str(seq + 1)+ '.npy'))
            labels_color = np.load(os.path.join(trainpath, 'labels' + str(seq + 1)+ '_color.npy'))
            labels_style = np.load(os.path.join(trainpath, 'labels' + str(seq + 1)+ '_style.npy'))
            labels_category = np.load(os.path.join(trainpath, 'labels' + str(seq + 1)+ '_category.npy'))
            print("---generate trainfile arrays",seq,"--", inputs.shape)
        start = pos*batch_size
        end = min((pos+1)*batch_size, set_len-1)
        batch_inputs = inputs[start:end]
        batch_labels_color = labels_color[start:end]
        batch_labels_style = labels_style[start:end]
        batch_labels_category = labels_category[start:end]
        pos += 1
        cnt += 1
        print("batch label shape ",batch_labels_color.shape, batch_labels_style.shape, batch_labels_category.shape)
        yield (batch_inputs, [batch_labels_color, batch_labels_style, batch_labels_category])

# Loading the data-------------------------------------------------------------
train_datagen = ImageDataGenerator(rotation_range=30.,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True)
test_datagen = ImageDataGenerator()

# 設定超參數HyperParameters
epochs = 50
batch = 128 #32
file_number = 8
file_len = 21600 #21600
x_val = np.load(os.path.join(data_path,'inputs1val.npy'))
y_val_category = np.load(os.path.join(data_path,'labels1val.npy'))
history = final_model.fit_generator(
    generate_arrays_from_file(data_path, file_len, file_number, batch_size=batch),
    # steps_per_epoch=(len(x_train) + batch_size - 1) // batch_size,
    steps_per_epoch=file_number * (file_len / batch),
    epochs=epochs,
    validation_data=(x_val, y_val_category),
    callbacks=[early_stopping, rlr]
    )

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title('res50_deepfashion2_cate_0204')
    plt.savefig(pltsave_path +'/res50_deepfashion2_cate_0204.png') 
    plt.show()


plot_learning_curves(history)
final_model.save('res50_deepfashion2_cate_0204.h5')
final_model.save(base_path +'/model/res50_deepfashion2_cate_0204.h5')