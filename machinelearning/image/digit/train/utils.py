#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 通用函数
from __future__ import print_function
import sys
import os
import logging
import time
import traceback
import platform
import subprocess
from PIL import Image
from PIL import ImageOps
import cv2
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
#公用static目录，通过http://www.yanjingang.com/static/tmp/1.txt 可访问过程文件
STATIC_PATH='/tmp/static/'

# 定义输入层及网络结构: 单层全连接层+softmax
def softmax_regression():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    return predict

# 定义输入层及网络结构: 多层感知器+relu*2+softmax(Multilayer Perceptron, MLP) 
def multilayer_perceptron():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # first fully-connected layer, using ReLu as its activation function
    hidden = fluid.layers.fc(input=img, size=128, act='relu')
    # second fully-connected layer, using ReLu as its activation function
    hidden = fluid.layers.fc(input=hidden, size=64, act='relu')
    # The thrid fully-connected layer, note that the hidden size should be 10,
    # which is the number of unique digits
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction

# 定义输入层及网络结构: 卷积神经网络(Convolutional Neural Network, CNN)
def convolutional_neural_network():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
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
def train_program():
    # 定义训练用label数据层
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # 定义网络结构
    # predict = softmax_regression() # uncomment for Softmax
    # predict = multilayer_perceptron() # uncomment for MLP
    predict = convolutional_neural_network()  # uncomment for LeNet5

    # 定义cost损失函数
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    # acc用于在迭代过程中print 
    acc = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, acc]

# 定义优化器
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)

# 自定义mnist数据集reader
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

# 自定义image目录文件列表reader
def image_reader_creator(img_path,height,width):
    def reader():
        imgs = os.listdir(img_path)
        for i in xrange(len(imgs)):
	    #imgs[i] = '0-5.png'
	    #print(imgs[i])
	    label = imgs[i].split('.')[0].split('-')[1]
	    image = load_image(img_path + imgs[i],width,height)
	    #print(img_path + imgs[i])
	    yield image[0][0],int(label)
 
    return reader

# 加载测试图片数据
def load_image(img_path,height,width,rotate=0,invert=False,sobel=False,ksize=5,dilate=0,erode=0,save_resize=False):
    if sobel: #边缘检测
	img_path = image_sobel(img_path, ksize=ksize, dilate=dilate, erode=erode)
    #加载图片
    im = Image.open(img_path).convert('L')
    #缩略图
    im = im.resize((height, width), Image.ANTIALIAS)
    #旋转
    if rotate != 0: #旋转度数
        im = im.rotate(rotate)
    #反转颜色(不要跟sobel一起用，因为sobel已经自动转为黑底+白边缘了)
    if invert:
        im = ImageOps.invert(im)
    #临时保存
    if save_resize:
        name = img_path.split('/')[-1]
        #resize_path = img_path.replace(name,'') + '../tmp/' + name.split('.')[0]+"_"+str(height)+"x"+str(width)+"."+name.split('.')[1]
        resize_path = STATIC_PATH + 'tmp/' + name.split('.')[0]+"_"+str(height)+"x"+str(width)+"."+name.split('.')[1]
        print(resize_path)
        im.save(resize_path)
    #返回数据
    im = numpy.array(im).reshape(1, 1, height, width).astype(numpy.float32)  #[N C H W] N几张图;C=1灰图;H高;W宽
    im = im / 255.0 * 2.0 - 1.0
    return im

def image_sobel(img_path, ksize=5, dilate=0, erode=0, dilate2=0):
    """图片边缘检测"""
    img = cv2.imread(img_path)
    #灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #write_image(gray, img_path, 'gray')
    # 高斯平滑
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    #write_image(gaussian, img_path, 'gaussion')
    # 中值滤波
    median = cv2.medianBlur(gaussian, 5)
    #write_image(median, img_path, 'median')
    # Sobel算子，X方向求梯度,对图像进行边缘检测
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=ksize) #ksize:1/3/5/7   cv2.CV_8U/cv2.CV_16S
    #sobel = cv2.Sobel(median, cv2.CV_16S, 1, 0, ksize=ksize) #ksize:1/3/5/7   cv2.CV_8U/cv2.CV_16S
    sobel = cv2.convertScaleAbs(sobel)
    # 二值化
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    threshold_path = write_image(binary, img_path, 'threshold')
    if dilate == 0 and erode == 0: 
	return threshold_path
    else:
        # 膨胀和腐蚀操作的核函数
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
        # 膨胀一次，让轮廓突出
        dilation = cv2.dilate(binary, element2, iterations=dilate) #iterations=1
        dilation_path = write_image(dilation, img_path, 'dilation')
	if erode > 0:# 腐蚀，去掉细节
            dilation = cv2.erode(dilation, element1, iterations=erode) #iterations=1
            dilation_path = write_image(dilation, img_path, 'erosion')
	if dilate2 > 0: # 再次膨胀，让轮廓明显一些
            dilation2 = cv2.dilate(erosion, element2, iterations=dilate2) #iterations=3设置太大但车牌区域很小时非车牌区域容易边缘连片过度，设置太小但车牌占比过大时容易省简称和后边连不上
            dilation_path = write_image(dilation2, img_path, 'dilation2')
        return dilation_path

def write_image(img, img_path, step='', path='tmp'):
    """保存图片并打印"""
    name = img_path.split('/')[-1]
    img_path = img_path.replace(name,'')
    #print(name)
    #print(img_path)
    if step != '':
        #img_file = img_path+'../'+path+'/'+name.split('.')[0]+'_'+step+'.'+name.split('.')[1]
        img_file = STATIC_PATH+path+'/'+name.split('.')[0]+'_'+step+'.'+name.split('.')[1]
    else:
        #img_file = img_path+'../'+path+'/'+name
        img_file = STATIC_PATH+path+'/'+name
    cv2.imwrite(img_file, img)
    print(img_file)
    return img_file

def mkdir(path):
    """检查并创建目录"""
    if not os.path.exists(path):
        os.makedirs(path)

LOG_LEVEL = logging.DEBUG
def log(type, msg, model=''):
    """写日志"""
    if type == 'debug' and LOG_LEVEL != logging.DEBUG:  # 线上配置日志等级不是DEBUG时跳过debug日志
        return

    try:
        model = model.__name__.lower()
    except:
        pass
        #model = type
    curtime = time.time()
    millisecond = int((curtime - int(curtime)) * 10000)
    fo = open("./log/" + model + "." + time.strftime('%Y%m%d%H',time.localtime(curtime)) + ".log","a")
    if isinstance(msg, unicode):
        msg = msg.encode("utf8")
    elif isinstance(msg, str) == False:
        msg = str(msg)
    msg = '[' + type.upper() + ']\t' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(curtime)) + "." + "%04d" % millisecond + "\t" + msg
    fo.write(msg + "\n")
    fo.close()

    if LOG_LEVEL == logging.DEBUG:  # debug时输出到屏幕
        print(msg)

def debug(msg, model=''):
    """
    debug日志打印
    :param msg:
    :param model:
    :return:
    """
    log('debug', msg, model)

def info(msg, model=''):
    """
    info日志打印
    :param msg:
    :param model:
    :return:
    """
    log('info', msg, model)

def warning(msg, model=''):
    """
    warning日志打印
    :param msg:
    :param model:
    :return:
    """
    log('warning', msg, model)

def error(msg, model=''):
    """
    error日志打印
    :param msg:
    :param model:
    :return:
    """
    log('error', msg, model)

def get_trace():
    """获得异常栈内容"""
    try:
        errmsg = "Traceback (most recent call last):\n "
        exc_type, exc_value, exc_tb = sys.exc_info()
        for filename, linenum, funcname, source in traceback.extract_tb(exc_tb):
            errmsg += "  File \"%-23s\", line %s, in %s() \n\t  %s \n" % (filename, linenum, funcname, source)
        errmsg += str(exc_type.__name__) + ": " + str(exc_value)
        # traceback.print_exc()
    except:
        #traceback.print_exc()
        errmsg = ''
    return errmsg



if __name__ == '__main__':
    #函数测试
    info("test","test")
    warning("test","test")
    print(1)
    """
    mnist_path = './data/mnist/'
    train_image   = mnist_path + 'train-images-idx3-ubyte.gz'
    train_label   = mnist_path + 'train-labels-idx1-ubyte.gz'
    for img,label in mnist_reader_creator(train_image,train_label,1)():  #reader
        print(img)
        print(len(img))
        print(len(img[0]))
        print(label)
	break
                             
    img_path = './data/train/'
    for img,label in image_reader_creator(img_path,28,28)():  #reader
        print(img)
        print(len(img))
        print(len(img[0]))
        print(label)
	break
    """                          
