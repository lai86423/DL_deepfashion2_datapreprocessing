import colorList
import  cv2
import numpy as np
import collections


#兩個顏色辨識 測試
maxsum = -100
i = 0
color = [10,5,8,18,55,18,5,3,10,2,0,1,65]
color_area = {}
color_dict = colorList.getColorList()

for d in color_dict:
    i +=1
    color_area[color[i]] = d

color_area2 = collections.OrderedDict(sorted(color_area.items()))

color_area2 = {v:k for k,v in color_area2.items()}
big = list(color_area2.items())[-2:]
diff = (big[1][1] - big[0][1])/ sum(color_area2.values())
if diff <=0.1 and (big[1][1]/ sum(color_area2.values())) >= 0.3:
    print("Diff",diff)
print(color_area,color_area2,big[0],big[1], big[0][1]/sum(color_area2.values()) ) 