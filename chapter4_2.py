# -*- coding: utf-8 -*-
# @File    : chapter4_2.py
# @Date    : 2020-05-25
# @Author  : fengluoluo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#在内存中生成模拟数据
# def GenerateData(training_epochs,batchSize=100):
#     for i in range(training_epochs):
#         train_X = np.linspace(-1,1,batchSize)
#         train_Y = 2 * train_X + np.random.randn(*train_X.shape)*0.3
#         yield shuffle(train_X,train_Y),i
#
# Xinput = tf.placeholder("float",(None))
# Yinput = tf.placeholder("float",(None))
#
# training_epochs = 20
#
# with tf.Session() as sess:
#     for (x,y),li in GenerateData(training_epochs):
#         xv,yv = sess.run([Xinput,Yinput],feed_dict={Xinput:x,Yinput:y})
#         print(li,"x.shape",np.shape(xv),"|x[:3]:",xv[:3])
#         print(li,"|y.shape:",np.shape(yv),"|y[:3]",yv[:3])
#
# #显示模拟数据
# train_data = list(GenerateData(1))[0]
# plt.plot(train_data[0][0],train_data[0][1],'ro',label="Original data")
# plt.legend()
# plt.show()
a = [1,2,3,4,5]
print(*a)