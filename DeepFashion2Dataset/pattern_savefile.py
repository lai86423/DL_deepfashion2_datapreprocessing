import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import re


base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
val_path = base_path + '/validation'


pattern_path ='/home/irene/deepfashion2/DeepFashion2Dataset/img_pattern/'
pattern_x_dir = train_path + '/pattern_x.txt'
pattern_y_dir = train_path + '/pattern_y.txt'
pattern_x_file = open(pattern_x_dir,'w')
pattern_y_file = open(pattern_y_dir,'w')

def pattern(name, label):
    pattern_x_file.write(name+'\n')
    pattern_y_file.write(label+'\n')

def img_pattern_file():
    for files in os.listdir(pattern_path):
        print(files)
        if files == 'text':
            for filename in os.listdir(pattern_path+'text'):
                pattern(filename,'1')
        elif files == 'dotted':
            for filename in os.listdir(pattern_path+'dotted'):
                pattern(filename,'2')
        elif files == 'checkered':
            for filename in os.listdir(pattern_path+'checkered'):
                pattern(filename,'3')
        elif files == 'striped':
            for filename in os.listdir(pattern_path+'striped'):
                pattern(filename,'4')
        elif files == 'pattern':
            for filename in os.listdir(pattern_path+'pattern'):
                pattern(filename,'5')
        elif files == 'solid':
            for filename in os.listdir(pattern_path+'solid'):
                pattern(filename,'6')

                
                    


img_pattern_file()