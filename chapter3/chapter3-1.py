# -*- coding: utf-8 -*-
# @File    : chapter3-1.py
# @Date    : 2020-05-15
# @Author  : fengluoluo
import sys
nets_path = r'./slim'
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print("already add slim")
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from nets.nasnet import pnasnet
import numpy as np
from  datasets import imagenet
from tensorflow.contrib import slim

tf.reset_default_graph()

image_size = pnasnet.build_pnasnet_large.default_image_size #获得图片输入尺寸
labels = imagenet.create_readable_names_for_imagenet_labels()#获得数据集标签
print(len(labels),labels)

def getone(onestr):
    return onestr.replace(","," ")

with open("中文标签.csv",'r+') as f: #打开文件
    labels = list(map(getone,list(f)))
    print(len(labels),type(labels),labels[:5])#显示输出中文标签

sample_images = ["hy.jpg","ps.jpg" ,"72.jpg"] #定义待测试图片路径

input_imgs = tf.placeholder(tf.float32,[None,image_size,image_size,3])#定义占位符
x1 = 2*(input_imgs/255.0)-1.0#归一化图片
arg_scope = pnasnet.pnasnet_large_arg_scope() #获取模型命名空间

with slim.arg_scope(arg_scope):
    logits,end_points = ...

























