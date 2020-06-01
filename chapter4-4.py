# -*- coding: utf-8 -*-
# @File    : chapter4-4.py
# @Date    : 2020-05-26
# @Author  : fengluoluo
#csv文件处理
"""
tf.train.start_input_producer
tf.reader.read_file
tf.decode_csv bmp
tf.train.batch作者shuffle batch
sess
coordinate
thread
"""


import tensorflow as tf

def read_data(file_queue): #处理CSV文件函数
    reader = tf.TextLineReader(skip_header_lines=1) #tf.TextLineReader
    key,value = reader.read(file_queue)
    defaults = [[0], [0.], [0.], [0.], [0.], [0]]       #为每个字段设置初始值
    csvcolumn = tf.decode_csv(value,defaults) #tf.decode_csv对每一行进行解析

    featurecolumn = [i for i in csvcolumn[1:-1]] #分类出类中的样本数据列
    labelcolumn = csvcolumn[-1]  #分离出列中的标签数据列

    return tf.stack(featurecolumn),labelcolumn #返回结果

def create_pipeline(filename,batch_size,num_epochs=None):
    #创建一个输入队列
    file_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)
    feature,label = read_data(file_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size

    feature_batch,label_batch = tf.train.shuffle_batch(
        [feature,label],batch_size=batch_size,capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return feature_batch,label_batch


#读取训练集
x_train_batch,y_train_batch = create_pipeline("iris_training.csv",32,num_epochs=100)
#读取测试集
x_test,y_test = create_pipeline("iris_test.csv",32)

with tf.Session() as sess:

    init_op = tf.global_variables_initializer()                 #初始化
    local_init_op = tf.local_variables_initializer()            #初始化本地变量，没有回报错
    sess.run(init_op)
    sess.run(local_init_op)

    coord = tf.train.Coordinator()                          #创建协调器
    threads = tf.train.start_queue_runners(coord=coord)    #开启线程列队

    try:
        while True:
            if coord.should_stop():
                break
            example, label = sess.run([x_train_batch, y_train_batch]) #注入训练数据
            print ("训练数据：",example) #打印数据
            print ("训练标签：",label) #打印标签
    except tf.errors.OutOfRangeError:       #定义取完数据的异常处理
        print ('Done reading')
        example, label = sess.run([x_test, y_test]) #注入测试数据
        print ("测试数据：",example) #打印数据
        print ("测试标签：",label) #打印标签
    except KeyboardInterrupt:               #定义按ctrl+c键时，对应的异常处理
        print("程序终止...")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()