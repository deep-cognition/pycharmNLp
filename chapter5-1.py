# -*- coding: utf-8 -*-
# @File    : chapter5-1.py
# @Date    : 2020-05-15
# @Author  : fengluoluo
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import *
import numpy as np

# dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
# dataset = dataset.range(5)
# iterator = dataset.make_one_shot_iterator()
# one = iterator.get_next()
# with tf.Session() as sess:
#     for i in range(5):
#         print(sess.run(one))

dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset =  Dataset.zip((dataset1,dataset2))
myiter = dataset.make_one_shot_iterator()
one = myiter.get_next()
with tf.Session() as sess:

   print(sess.run(one))

