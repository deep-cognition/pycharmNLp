# -*- coding: utf-8 -*-
# @File    : chapter4_3.py
# @Date    : 2020-05-25
# @Author  : fengluoluo
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
def load_sample(sample_dir):
    """
    递归读取文件，
    :param sample_dir:
    :return:
    """
    print("loading sample dataset...")
    lfilename = []
    labelsample = []
    for (dirpath,dirnames,filenames) in os.walk(sample_dir): #递归遍历文件夹
        for filename in filenames:
            filename_path = os.sep.join([dirpath,filename])
            lfilename.append(filename_path)
            labelsample.append(dirpath.split("\\")[-1])
    lab = list(sorted(set(labelsample)))
    labeldict = dict(zip(lab.list(range(len(lab)))  ))#生成字典
    labels = [labeldict[i] for i in labelsample]
    return shuffle(np.asarray(lfilename),np.asarray(labels)),np.asarray(lab)
data_dir = "mnist_digits_images"
#zip进行字典的创建，然后进行标签的数字转化
(image,label),labelsnames = load_sample(data_dir)   #载入文件名称与标签
print(len(image),image[:2],len(label),label[:2])#输出load_sample返回的数据结果
print(labelsnames[ label[:2] ],labelsnames)#输出load_sample返回的标签字符串

def get_batches(image,label,input_w,input_h,channels,batch_size):
    queue = tf.train.slice_input_producer([image,label]) #实现一个输入队列
    label = queue[1] #从输入队列里读取标签
    image_c = tf.read_file(queue[0])#从输入队列里读取image路径
    image = tf.image.decode_bmp(image_c,channels)
    image = tf.image.resize_image_with_crop_or_pad(image,input_w,input_h)
    image = tf.image.per_image_standardization(image)

    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size=batch_size,
                                             num_threads=64) #
    image_batch = tf.cast(image_batch,tf.float32) #将数据类型转换为float32
    label_batch = tf.reshape(label_batch,[batch_size])



