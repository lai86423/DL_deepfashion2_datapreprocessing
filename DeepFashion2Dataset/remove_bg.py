#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np 
import os
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
img_dir = base_path + '/train/img_hand/'
new_img_dir = base_path + '/train/img_hand_new/'
#filename='003147.jpg'

def cutimgBg(frame):
    pos =[]
    smask =[]
    color = [0,0,0]
    #try:
    smask = np.all(frame != color,axis=2)

    pos = np.where(smask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])

    frame2 = frame[ymin: ymax, xmin: xmax]
    a = 7
    y_fix = int((1/a)*np.shape(frame2)[0])
    x_fix = int((1/a)*np.shape(frame2)[1])
    #print(y_fix,x_fix)
    new_img = frame2[y_fix: y_fix*(a-1), x_fix: x_fix*(a-1)]

    #except:
    #    new_img = frame
    return new_img


def cutBg():
    for filename in os.listdir(img_dir):
        pos =[]
        smask =[]
        frame = cv2.imread(img_dir+filename)
        color = [0,0,0]
        #print(frame)
        try:
            smask = np.all(frame != color,axis=2)
            pos = np.where(smask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            new_img = frame[ymin:ymax, xmin:xmax]
        except:
            print(filename)
            new_img = frame
        cv2.imwrite(new_img_dir + filename, new_img)

allfile = os.listdir(img_dir)
allnewfile = os.listdir(new_img_dir)
print(len(allfile),len(allnewfile))