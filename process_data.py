# -*- coding: utf-8 -*-
# @File    : process_data.py
# @Date    : 2020-05-29
# @Author  : fengluoluo
import json
import cv2
import os
dirname = "./json"
count = 1
for filename in os.listdir("./json"):
    with open(os.path.join(dirname,filename)) as f:
        data = json.load(f)
    shapes = data['shapes']
    img = cv2.imread(data["imagePath"])
    for points in shapes:
        point_1_x,point_1_y,point_2_x,point_2_y = points['points'][0][0],points['points'][0][1],points['points'][1][0],points['points'][1][1]

