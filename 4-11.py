# -*- coding: utf-8 -*-
# @File    : 4-11.py
# @Date    : 2020-05-26
# @Author  : fengluolu
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def dataset(directory,size,batchsize):#定义函数
    """parse dataset"""
    def _parseone(example_proto): #解析一个图片文件
        """reading and handle image"""
        #定义解析的字典
        dics={}
        dics["label"]= tf.FixedLenFeature(shape=[],dtype=tf.int64)
        dics['img_raw'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
        parsed_example = tf.parse_single_example(example_proto,dics)#解析一行样本
        image = tf.decode_raw(parsed_example['img_raw'],out_type=tf.uint8)
        image = tf.reshape(image,size)
        image = tf.cast(image,tf.float32)*(1./255)-0.5 #对图像数据做归一化

        label = parsed_example['label']
        label = tf.cast(label,tf.int32)
        label = tf.one_hot(label,depth=2,on_value=1)
        return image,label

    dataset = tf.data.TFRecordDataset(directory)
    dataset = dataset.map(_parseone)
    dataset = dataset.batch(batchsize)

    dataset = dataset.prefetch(batchsize)
    return dataset

def showimg(index,label,img,ntop):#显示
    plt.figure(figsize=(20,10))#显示图片的宽、高
    plt.axis("off")
    ntop = min(ntop,9)
    print(index)
    for i in range(ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()

def showresult(subplot,title,thisimg):
    p=plt.subplot(subplot)
    p.axis("off")
    p.imshow(thisimg)
    p.set_title(title)


def getone(dataset):
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    return one_element

sample_dir = ['mydata.tfrecords']
size = [256,256,3]
batchsize = 10
tdataset = dataset(sample_dir,size,batchsize)
# print(tdataset.output_types)
# print(tdataset.output_shapes)

one_element1 = getone(tdataset)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#初始化
    try:
        for step in np.arange(1):
            value = sess.run(one_element1)
            showimg(step,value[1],np.array((value[0]+0.5)*225,np.uint8),10)
    except tf.errors.OutOfRangeError:
        print("Done!")
