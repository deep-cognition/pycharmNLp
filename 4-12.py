# -*- coding: utf-8 -*-
# @File    : 4-12.py
# @Date    : 2020-05-29
# @Author  : fengluoluo
import os
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

#开启动态图的模式
tf.enable_eager_execution()
print("Tensorflow 版本:{}".format(tf.__version__))
print("Eager execution:{}".format(tf.executing_eagerly()))


#os.walk进行两级目录的结构
def load_sample(sample_dir,shuffleflag=True):
    print('loading sample dataset ...')
    lfilenames = []
    labelsnames = []
    for (dirpath,dirnames,filenames) in os.walk(sample_dir):#递归遍历文件夹
        for filename in filenames:
            filename_path = os.sep.join([dirpath,filename])
            lfilenames.append(filename_path)
            labelsnames.append(dirpath.split("\\")[-1])#添加文件名对应的标签
    lab = list(sorted(set(labelsnames))) #生成标签名称列表
    labeldict = dict(zip(lab,list(range(len(lab)))))
    labels = [labeldict[i] for i in labelsnames]
    if shuffleflag:
        return shuffle(np.asarray(lfilenames),np.asarray(labels)),np.asarray(lab)
    else:
        return (np.asarray(lfilenames),np.asarray(labels)),np.asarray(lab)

directory = 'man_woman'
(filenames,labels),_ = load_sample(directory,shuffleflag=False)


def dataset(directory,size,batchsize,random_rotated=False):#定义函数，创建数据集
    """parse dataset"""
    (filenames,labels),_ = load_sample(directory,shuffleflag=False)

