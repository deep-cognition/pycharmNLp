4# -*- coding: utf-8 -*-
# @File    : chapter4-1.py
# @Date    : 2020-05-15
# @Author  : fengluoluo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#在内存中生成模拟数据
def generateData(batchsize=128):
    train_X = np.linspace(-1,1,batchsize)
    train_Y = 2* train_X+np.random.randn(*train_X.shape)*0.3
    yield train_X,train_Y #以生成器的方式进行返回

#定义网络模型结构部分，这里只有占位符张量
Xinput = tf.placeholder("float",(None));
Yinput = tf.placeholder("float",(None));

training_epoch =20
with tf.Session() as sess:#建立会话
    for epoch in range(training_epoch):
        for x,y in generateData():
            xv,yv = sess.run([Xinput,Yinput],feed_dict={Xinput:x,Yinput:y})