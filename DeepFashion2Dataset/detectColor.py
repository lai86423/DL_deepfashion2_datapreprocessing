import  cv2
import numpy as np
import colorList
path ='/home/irene/deepfashion2/DeepFashion2Dataset/train/image_new/color/'

#處理圖片
def get_color(frame,name):
    #print('go in get_color')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = colorList.getColorList()
    for d in color_dict: #輪流計算顏色面積
        #切割出指定顏色區域
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1]) 
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
            color = d

    return color


if __name__ == '__main__':
    base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
    new_img_dir = base_path + '/train/image_new/'
    filename='000015_Up.jpg'
    frame = cv2.imread(new_img_dir+filename)
    print(get_color(frame,filename))