# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import re
#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
train_path = base_path + '/train'
train_top_dir_file = base_path + '/train/train_top_dir_file.txt'
train_top_label_file = base_path + '/train/train_top_label_file.txt'

val_path = base_path + '/validation'
val_label_dir = base_path + '/validation/annos'
val_top_dir_file = base_path + '/validation/val_dir_file.txt'
val_top_label_file = base_path + '/validation/val_label_file.txt'

new_img_dir = base_path + '/train/image_new/'
def ReadFile(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line[:-1])
    print(len(data)) 
    return data
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/irene/local/src/openpose/models/"
    params["face"] = True
    params["hand"] = True

    def count_limitPoint(space, percent, point1, point2, max, min):
        print(" space*percent", space*percent)
        limitpoint1 = point1 - space*percent
        limitpoint2 = point2 + space*percent
        print("L1,L2",limitpoint1,limitpoint2,min)
        if limitpoint1 < min or limitpoint1 > limitpoint2:
            limitpoint1 = min
        if limitpoint2 >max or limitpoint1 > limitpoint2:
            limitpoint2 = max

        return int(limitpoint1), int(limitpoint2)
        

    def openpose_preprocess(img_path,img_label, img_name, save_path,  new_img_list, new_img_label_list):
        # Flags
        #print(img_path)
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default=img_path, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item


        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        img = cv2.imread(args[0].image_path)
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        height, width, channels = img.shape

        #---判斷圖片有人
        if datum.poseKeypoints is not None: 
            a = np.zeros(3)
            #---判斷人包含頭則切掉
            # if (datum.poseKeypoints[0][0] == a).any() == False: 
            #     print("cut head!")
            #     # cut head 
            #     headH = datum.poseKeypoints[0][0][1] - datum.poseKeypoints[0][1][1]
            #     # 裁切圖片
            #     Y_nohead = int(datum.poseKeypoints[0][1][1] + headH*0.5)
            #     img_nohead = img[Y_nohead:height, 0:width]
            #     #cv2.imwrite(save_path + img_name+'_cuthead.jpg', img_nohead)
            #     height2, width, channels = img_nohead.shape
            # else:
            #     print("no head!")

            #---判斷有上身
            if (datum.poseKeypoints[0][1] == a).any() == False:
                if (datum.poseKeypoints[0][10] == a).any() == False : #and [(datum.poseKeypoints[0][13] == a).any() == False]   另隻腳先不判斷~
                    print("Have upper & lower")
                    #---切上身
                    try:
                        legH = datum.poseKeypoints[0][13][1] - datum.poseKeypoints[0][12][1]
                    except Exception as e:
                        print("legH",e)
                        legH = height /2    
                    #img_Up = img[int(datum.poseKeypoints[0][1][1]) : int(datum.poseKeypoints[0][8][1]) , 0:width]
                    try:
                        img_Up_height_N,  img_Up_height_S = count_limitPoint(legH, 0.1, int(datum.poseKeypoints[0][1][1]), int(datum.poseKeypoints[0][8][1]),height, 0 )
                    except Exception as e:
                        print("img_Up_height_N,  img_Up_height_S except",e)

                    #cv2.imwrite('output_up.jpg', img_up)
                    #height3, width, channels = img_Up.shape
                    # 切圖寬
                    try:
                        shoWid = datum.poseKeypoints[0][5][0] - datum.poseKeypoints[0][2][0]
                        print("shoWid", shoWid)
                    except Exception as e:
                        print(e)
                        shoWid = width/2
                    try:
                        img_Up_width_L,  img_Up_width_R= count_limitPoint(shoWid, 0.5, int(datum.poseKeypoints[0][3][0]),int(datum.poseKeypoints[0][6][0]),width, 0)

                    except Exception as e:
                        print("img_Up_width_L,  img_Up_width_R except",e)
                    # if shoWid > 0:
                    #     print("shoWid", shoWid)
                    #     try:
                    #         limit_L = int(datum.poseKeypoints[0][3][0]- 0.5*shoWid)
                    #         limit_R = int(datum.poseKeypoints[0][6][0] + 0.5*shoWid)
                    #        #print("limit_L:limit_R",limit_L,limit_R)
                    #         if(limit_L<0):
                    #             limit_L = 0
                    #         if(limit_R>width):
                    #             limit_R=width
                    print("N,S,L,R",img_Up_height_N,  img_Up_height_S,img_Up_width_L,  img_Up_width_R)
                    
                    img_Up = img[img_Up_height_N:img_Up_height_S,img_Up_width_L:img_Up_width_R]

                    print("img_label", int(img_label))
                    if int(img_label) <= 5:
                        cv2.imwrite(save_path + img_name+'_Up.jpg', img_Up)
                        new_img_list.append(save_path + img_name+'_Up.jpg')
                        new_img_label_list.append(img_label)
                    else:
                        cv2.imwrite(save_path +'img_nolabel/'+ img_name+'_Up.jpg', img_Up)
                        
                    #---切下身
                    try:
                        img_Down_height_N,  img_Down_height_S = count_limitPoint(legH, 1, int(datum.poseKeypoints[0][10][1]), int(datum.poseKeypoints[0][10][1]),height, int(datum.poseKeypoints[0][8][1])-0.1*legH )
                    except Exception as e:
                        print("img_Down_height_N,  img_Down_height_S except",e)
                    
                    try:
                        buttWid = datum.poseKeypoints[0][12][0] - datum.poseKeypoints[0][9][0]
                        print("buttWid", buttWid)
                    except Exception as e:
                        print("buttWid except",e)
                        buttWid = width/2
                    #img_Down = img_Down[0:height4, int(datum.poseKeypoints[0][9][0]- buttWid): int(datum.poseKeypoints[0][12][0] + buttWid)]
                    
                    try:
                        img_Down_width_L,  img_Down_width_R= count_limitPoint(buttWid, 1, int(datum.poseKeypoints[0][9][0]),int(datum.poseKeypoints[0][12][0]),width, 0)

                    except Exception as e:
                        print("img_Down_width_L,  img_Down_width_R except",e)
                
                    print("Down N,S,L,R",img_Down_height_N,  img_Down_height_S,img_Down_width_L,  img_Down_width_R)
                    
                    img_Down = img[img_Down_height_N:img_Down_height_S,img_Down_width_L:img_Down_width_R]


                    # try:
                    #     img_Down = img[int(datum.poseKeypoints[0][10][1]-legH) : int(datum.poseKeypoints[0][10][1]+legH), 0:width]
                    # except Exception as e:
                    #     print(e)
                    #     img_Down = img[int(datum.poseKeypoints[0][8][1]-0.1*legH) : height, 0:width]
                    # # #cv2.imwrite('output_down.jpg', img_down)
                    # height4, width, channels = img_Down.shape

                    # #切圖寬
                    # buttWid = datum.poseKeypoints[0][12][0] - datum.poseKeypoints[0][9][0]
                    # print("buttWid", buttWid)
                    # if buttWid > 0:
                    #     img_Down = img_Down[0:height4, int(datum.poseKeypoints[0][9][0]- buttWid): int(datum.poseKeypoints[0][12][0] + buttWid)]
                    
                    if int(img_label) > 5:
                        cv2.imwrite(save_path + img_name+'_Down.jpg', img_Down)
                        new_img_list.append(save_path + img_name+'_Down.jpg')
                        new_img_label_list.append(img_label)
                    else:
                        cv2.imwrite(save_path +'img_nolabel/'+ img_name+'_Down.jpg', img_Down)
                    

                #--只有上身  
                else:
                    print("Only upper")
                    #切圖寬
                    shoWid = datum.poseKeypoints[0][5][0] - datum.poseKeypoints[0][2][0]
                    print("shoWid", shoWid)
                    if shoWid > 0:
                        limit_L = int(datum.poseKeypoints[0][3][0]- 0.5*shoWid)
                        limit_R = int(datum.poseKeypoints[0][6][0] + 0.5*shoWid)
                        if(limit_L<0 or limit_L>width):
                            limit_L = 0
                        if(limit_R>width or limit_R<0):
                            limit_R = width
                    else:
                        limit_L = 0
                        limit_R = width

                    try:
                        img_Up = img[datum.poseKeypoints[0][1][1]:height, limit_L:limit_R]
                    except Exception as e:
                        print(e)
                        img_Up = img[0:height, limit_L:limit_R]
                    new_img_list.append(save_path + img_name+'_Up.jpg')
                    new_img_label_list.append(img_label)
                    cv2.imwrite(save_path + img_name+'_Up.jpg', img_Up)
            #--只有下身 
            else:
                print("Only lower")
                legH = datum.poseKeypoints[0][13][1] - datum.poseKeypoints[0][12][1]
                try:
                    img_Down = img[0 : int(datum.poseKeypoints[0][10][1]+legH), 0:width]
                except Exception as e:
                    print(e)
                    img_Down = img[0 : height, 0:width]
                #cv2.imwrite('output_down.jpg', img_Down)
                height4, width, channels = img_Down.shape

                #切圖寬 
                buttWid = datum.poseKeypoints[0][12][0] - datum.poseKeypoints[0][9][0]
                if buttWid > 0:
                    img_Down = img[0:height4, int(datum.poseKeypoints[0][9][0]- buttWid):int(datum.poseKeypoints[0][12][0] + buttWid)]
        
                cv2.imwrite(save_path + img_name+'_Down.jpg', img_Down)  
                new_img_list.append(save_path + img_name+'_Down.jpg')
                new_img_label_list.append(img_label)
        else:
            print("No human!")  
            cv2.imwrite(save_path + img_name+'.jpg', img) 
            new_img_list.append(save_path + img_name+'.jpg')
            new_img_label_list.append(img_label) 
        return new_img_list, new_img_label_list


    img_path1 = "/home/irene/local/src/openpose/examples/media/001385.jpg"

    img_list = []
    img_label_list = []
    old_img_file = ReadFile(train_top_dir_file)
    old_img_label_file = ReadFile(train_top_label_file)
    print(len(old_img_file),len(old_img_label_file))
    
    for i in range(13):
        i = i
        print(old_img_file[i], old_img_label_file[i])
        #print(img_list, img_label_list,str(i).zfill(6))
        img_list, img_label_list = openpose_preprocess(train_path+'/'+old_img_file[i], old_img_label_file[i],str(i+1).zfill(6), new_img_dir, img_list, img_label_list)
    print(img_list, img_label_list)    

except Exception as e:
        print(e)
        sys.exit(-1)
