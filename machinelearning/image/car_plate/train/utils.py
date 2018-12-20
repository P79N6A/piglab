#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import collections
import platform
import subprocess
from PIL import Image
import numpy
import paddle
import paddle.fluid as fluid
try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print(
        "In the fluid 1.0, the trainer and inferencer are moving to paddle.fluid.contrib",
        file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *


# 定义输入层及网络结构: 单层全连接层+softmax
def softmax_regression(width,height):
    img = fluid.layers.data(name='img', shape=[1, width, height], dtype='float32')
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    return predict

# 定义输入层及网络结构: 多层感知器+relu*2+softmax(Multilayer Perceptron, MLP) 
def multilayer_perceptron(width,height):
    img = fluid.layers.data(name='img', shape=[1, width, height], dtype='float32')
    # first fully-connected layer, using ReLu as its activation function
    hidden = fluid.layers.fc(input=img, size=128, act='relu')
    # second fully-connected layer, using ReLu as its activation function
    hidden = fluid.layers.fc(input=hidden, size=64, act='relu')
    # The thrid fully-connected layer, note that the hidden size should be 10,
    # which is the number of unique digits
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction

# 定义输入层及网络结构: 卷积神经网络(Convolutional Neural Network, CNN)
def convolutional_neural_network(width,height):
    img = fluid.layers.data(name='img', shape=[1, width, height], dtype='float32')
    # first conv pool
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # second conv pool
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # output layer with softmax activation function. size = 10 since there are only 10 possible digits.
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


# 定义训练损失函数 
def train_program(label_len=7):
    # 定义训练用label数据层
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    #label = fluid.layers.data(name='label', shape=[1, None, label_len], dtype='int64')

    # 定义网络结构
    # predict = softmax_regression() # uncomment for Softmax
    # predict = multilayer_perceptron() # uncomment for MLP
    #predict = convolutional_neural_network(60,180)  # uncomment for LeNet5
    predict = convolutional_neural_network(28,28)  # uncomment for LeNet5

    # 定义cost损失函数
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    # acc用于在迭代过程中print 
    acc = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, acc]

# 定义优化器
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)

# 自定义mnist图像数据集reader
def mnist_reader_creator(image_filename,label_filename,buffer_size):
    def reader():
        #调用命令读取文件，Linux下使用zcat
        if platform.system()=='Linux':
            zcat_cmd = 'zcat'
        else:
            raise NotImplementedError("This program is suported on Linux,\
                                      but your platform is" + platform.system())
        
        # 读取mnist图片集
        sub_img = subprocess.Popen([zcat_cmd, image_filename], stdout = subprocess.PIPE)
        sub_img.stdout.read(16) # 跳过前16个magic字节

        # 读取mnist标签集
        sub_lab = subprocess.Popen([zcat_cmd, label_filename], stdout = subprocess.PIPE)
        sub_lab.stdout.read(8)  # 跳过前8个magic字节
        
    	try:
            while True:         #前面使用try,故若再读取过程中遇到结束则会退出
		# 批量读取label，每个label占1个字节
                labels = numpy.fromfile(sub_lab.stdout,'ubyte',count=buffer_size).astype("int")
                if labels.size != buffer_size:
                    break
		# 批量读取image，每个image占28*28个字节，并转换为28*28的二维float数组
                images = numpy.fromfile(sub_img.stdout,'ubyte',count=buffer_size * 28 * 28).reshape(buffer_size, 28, 28).astype("float32")
        	# 像素值映射到(-1,1)范围内用于训练
                images = images / 255.0 * 2.0 - 1.0;
                for i in xrange(buffer_size):
                    yield images[i,:],int(labels[i]) #将图像与标签抛出，循序与feed_order对应！
        finally:
            try:
        	#结束img读取进程
                sub_img.terminate()
            except:
                pass
            try:
        	#结束label读取进程
                sub_lab.terminate()
            except:
                pass
    return reader

# 自定义images图像数据集reader
def image_reader_creator(img_path,width,height,label_idx):
    def reader():
        UNK_ID = label_idx['<unk>']
        # 读取image图片列表
        imgs = os.listdir(img_path)
        for name in imgs:
	    #print(img_path+name)
	    images = load_image(img_path+name, width, height)
	    #print(images[0][0][0])
	    image = images[0][0]
            label = []
	    label = [label_idx.get(c, UNK_ID) for c in name.split('.')[0].decode("utf8")]
	    #print(image)
	    #print(label)
            yield image,label[0] #将图像与标签抛出
    return reader

# 加载测试图片数据
def load_image(file,width,height):
    im = Image.open(file).convert('L')
    im = im.resize((width, height), Image.ANTIALIAS)  #缩放
    im = numpy.array(im).reshape(1, 1, width, height).astype(numpy.float32)  #[N C H W] N几张图;C=1灰图;H高;W宽
    im = im / 255.0 * 2.0 - 1.0  # 像素值映射到(-1,1)范围内用于训练
    return im

def image_label_dict(img_path, cutoff=0):
    """根据目录内的图片文件名生成label词典"""
    label_freq = collections.defaultdict(int)
    imgs = os.listdir(img_path)
    for name in imgs:
	#print(name.split('.')[0])
	for w in name.split('.')[0].decode("utf8"): #按字符编码
            label_freq[w.encode('utf8')] += 1
	    #print(w.encode('utf8'))

    # 筛选出现次数>cutoff次的词
    label_freq  = filter(lambda x: x[1] > cutoff, label_freq.items())
    #print(label_freq)
    # 按词计数倒排
    dictionary = sorted(label_freq, key=lambda x: (-x[1], x[0]))
    # 解压为二维矩阵，提取出词列表
    words, _   = list(zip(*dictionary))
    #print words, _
    # 给词表分配序号ID {'worda':1,'wordb':2,'wordc':3,'word...':..}
    word_idx   = dict(zip(words, xrange(len(words))))
    # 保存词总数
    word_idx['<unk>'] = len(words)
    #print(word_idx)
    return word_idx

if __name__ == '__main__':
    #函数测试
    img_path = '/home/work/paddle/sample/car_platenumber/train/data/images_train/'
    label_idx = image_label_dict(img_path)  #label词典
    for img,label in image_reader_creator(img_path,60,180,label_idx)():  #reader
	print(img)
        print(len(img))
        print(len(img[0]))
        print(label)
        break


