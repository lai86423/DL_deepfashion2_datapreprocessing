import  cv2
import numpy as np
import colorList
import os
import remove_bg
#處理圖片
def get_color(frame,name):
    #print('go in get_color')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color_area = {}
    color = None
    color_dict = colorList.getColorList()
    for d in color_dict: #輪流計算顏色面積
        #切割出指定顏色區域
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1]) 
        #if d =='skin' or d == 'orange':
            #print(d)
            #cv2.imwrite(path+name+d+'.jpg',mask)
        #影像二值化
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        #影像膨脹
        binary = cv2.dilate(binary,None,iterations=2)
        #輪廓檢測 
        cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            #輪廓面積
            sum+=cv2.contourArea(c)
        if sum > maxsum :
            #最多面積之顏色 判斷為該顏色
            maxsum = sum
            color = d[:1]

    return color


if __name__ == '__main__':
    base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
    train_path = base_path + '/train'    
    img_dir = train_path+'/img_body/'
    new_img_dir = train_path+'/img_body_small/'
    AllFile = os.listdir(img_dir)
    for filename in AllFile: 
        if filename.endswith(".png") or filename.endswith(".jpg"):
            try:
                frame = cv2.imread(img_dir+filename) 
                new_frame = remove_bg.cutimgBg(frame)
                cv2.imwrite(new_img_dir + get_color(new_frame,filename) + '_' +filename , new_frame)
            except:
                print("no")
                pass