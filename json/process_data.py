# -*- coding: utf-8 -*-
# @File    : process_data.py
# @Date    : 2020-05-29
# @Author  : fengluoluo
import json
import cv2
import os
dirname = "../train_label"
count = 1
for filename in os.listdir("../train_label"):
    with open(os.path.join(dirname,filename)) as f:
        data = json.load(f)
    shapes = data['shapes']
    path = data["imagePath"].split("\\")[-1]
    img = cv2.imread(os.path.join("../train",path))
    for points in shapes:
        point_1_x,point_1_y,point_2_x,point_2_y = int(points['points'][0][0]),\
                                                  int(points['points'][0][1]),\
                                                  int(points['points'][1][0]),\
                                                  int(points['points'][1][1])
        label = points['label']
        region = img[point_1_y:point_2_y,point_1_x:point_2_x]
        if not os.path.exists("../images/"+label):
            os.mkdir("../images/"+label)
        cv2.imwrite("../images/{}/{}.jpg".format(label,count),region)
        count = count + 1
