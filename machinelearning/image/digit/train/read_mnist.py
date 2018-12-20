#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 读取mnist数据集的图片和label，并生成以序号+label命名的图片
# cmd: nohup python read_mnist.py >log/read_mnist.log &
import os
import random
import platform
import numpy
import subprocess
from PIL import Image
import utils

#mnist训练集目录
mnist_path = './data/mnist/'
#mnist_path = '~/.cache/paddle/dataset/mnist/'
train_image   = mnist_path + 'train-images-idx3-ubyte.gz'
train_label   = mnist_path + 'train-labels-idx1-ubyte.gz'
test_image    = mnist_path + 't10k-images-idx3-ubyte.gz'
test_label    = mnist_path + 't10k-labels-idx1-ubyte.gz'

#读取出的图片存放位置
output_path = './data/'


def reader_mnist(image_filename,label_filename,buffer_size=200,path='train'):
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
        id = 0 #图片集序号
        while True:         #前面使用try,故若再读取过程中遇到结束则会退出
            # 批量读取label，每个label占1个字节
            labels = numpy.fromfile(sub_lab.stdout,'ubyte',count=buffer_size).astype("int")
            if labels.size != buffer_size:
                break
            # 批量读取image，每个image占28*28个字节，并转换为28*28的二维float数组
            images = numpy.fromfile(sub_img.stdout,'ubyte',count=buffer_size * 28 * 28).reshape(buffer_size, 28, 28).astype("float32")
            for i in xrange(buffer_size):
                id += 1
                img = images[i]
		num = labels[i]
                #print img
                #print num
                #创建新28*28图片对象
                image = Image.new('L', (28, 28))
                for x in xrange(28):
                    for y in xrange(28):
		        #print img[x][y]
                        image.putpixel((y, x), int(img[x][y])) #按像素写入

                #保存图片(序号-label.png)
		utils.mkdir(output_path + path)
		save_file = output_path + path + '/' + str(id) + '-' + str(num) + '.png'
		image.save(save_file) 
		print save_file 
                #break
	    #break

    finally: 
	#结束读取进程
        sub_img.terminate()
        sub_lab.terminate()


if __name__ == '__main__':
    #读取训练集
    #reader_mnist(train_image,train_label,buffer_size=200,path='train')
    #读取测试集
    reader_mnist(test_image,test_label,buffer_size=200,path='test')
