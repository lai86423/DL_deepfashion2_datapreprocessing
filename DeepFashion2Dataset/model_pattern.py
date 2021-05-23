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
import tensorflow as tf
from PIL import Image
import efficientnet.keras as efn 

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=1,
                        mode='max',
                        epsilon=0.0001)

#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
trainpath = base_path + '/train'
val_path = base_path + '/validation'
pltsave_path = base_path+'/plt_img'

# Model -----------------------------------------------------------------------
model_net = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model_net = efn.EfficientNetL2(input_shape=(128,128,3), # 當 include_top=False 時，可調整輸入圖片的尺寸（長寬需不小於 32）
  weights="./efficientnet-l2_noisy-student_notop.h5", 
  #weights='imagenet',
  include_top=False,# 是否包含最後的全連接層 (fully-connected layer)
  drop_connect_rate=0,  # the hack
  pooling='avg'# 當 include_top=False 時，最後的輸出是否 pooling（可選 'avg' 或 'max'）
)
#model_net = tf.keras.applications.EfficientNetB7(input_shape=(128,128,3),weights="imagenet",include_top=False, pooling='avg',classifier_activation="softmax")
#freeze some layers
#for layer in model_net.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    #layer.trainable = False
#model_net.trainable = False

#build the category classification branch in the model
x = model_net.output
x = layers.Dropout(0.5)(x)
#category
#5/10 神經元從512顆改為4試看看
x1 = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
y1 = Dense(4, activation='softmax', name='category')(x1)

#create final model by specifying the input and outputs for the branches
final_model = Model(inputs=model_net.input, outputs=y1)

print(final_model.summary())

#opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
opt = Adam(learning_rate=0.001)

final_model.compile(optimizer=opt,loss={'category':'categorical_crossentropy'
                                        },
                    metrics={'category':['accuracy'] 
         }
                    ) #default:top-5

# Loading the data-------------------------------------------------------------

train_datagen = ImageDataGenerator(#rotation_range=30.,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    brightness_range= [0.8, 1.2],
                                    rotation_range = 5,
                                    channel_shift_range=100
                                    #preprocessing_function = myFunc
                                    #vertical_flip = True
                                    )

test_datagen = ImageDataGenerator()

def generate_arrays_from_file(trainpath,set_len,file_nums,has_remainder=0,batch_size=32):
    
    cnt = 0 
    pos = 0
    inputs = None
    labels_category = None

    while 1:
        if cnt % (set_len // batch_size+has_remainder) == 0:  #?�斷?�否讀完�??�個�?�?
            pos = 0
            seq = cnt // (set_len // batch_size + has_remainder) % file_nums #此次讀?�第seq?��?�?
            del inputs, labels_category
            inputs = np.load(os.path.join(trainpath, 'inputs' + str(seq + 1)+ '0420_train_sleeve.npy'))
            labels_category = np.load(os.path.join(trainpath, 'labels' + str(seq + 1)+ '0420_train_sleeve.npy'))
        print("---generate trainfile arrays",seq,"--", inputs.shape)
        start = pos*batch_size
        end = min((pos+1)*batch_size, set_len-1)
        batch_inputs = inputs[start:end]
        batch_labels_category = labels_category[start:end]
        pos += 1
        cnt += 1
        print("batch label shape ",batch_labels_category.shape)
        yield (batch_inputs, batch_labels_category)

# 設�?超�??�HyperParameters
epochs = 300
batch = 32 #128
file_number = 1
file_len = 3776#21600
x_val = np.load(os.path.join(val_path,'inputs1val_down_0509_clean.npy'))
y_val_category = np.load(os.path.join(val_path,'labels1val_down_0509_clean.npy'))

x_train = np.load(os.path.join(trainpath,'inputs1train_down_0509_clean.npy'))
y_train = np.load(os.path.join(trainpath,'labels1train_down_0509_clean.npy'))

train_generator = train_datagen.flow(
    x_train,
    y=y_train,
    batch_size=batch,
    shuffle=True,
)

history = final_model.fit_generator(
    #generate_arrays_from_file(trainpath, file_len, file_number, batch_size=batch),
    train_generator,
    #steps_per_epoch=file_number * (file_len / batch),
    epochs=epochs,
    validation_data=(x_val, y_val_category),
    callbacks=[early_stopping, rlr]
    )

name ='EfficientNetL2_down_0519_rotate5'

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.5)
    plt.title(name)
    plt.savefig(pltsave_path +'/'+ name +'.png') 
    plt.show()

plot_learning_curves(history)
final_model.save(base_path +'/model/+'+ name +'.h5')