#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 手写数字识别模型测试
#cmd: python infer.py infer_62.jpeg
from __future__ import print_function
import sys
import os
import platform
import subprocess
from PIL import Image
from PIL import ImageOps
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
import utils
import getopt

def infer(img_file='',model_path='./'):
    if img_file == '':
        opts, args = getopt.getopt(sys.argv[1:], "p:", ["file_name="])
        if len(args) == 0:
            print("usage:  python infer.py [file_name_name]  \n\tpython infer.py infer_62.jpeg")
            return 1,'file_name is empty',{}
        file_name = args[0]
        img_file = './data/image/' + file_name
    else:
        file_name = img_file.split('/')[-1] 

    # 是否使用GPU
    use_cuda = False  # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    params_dirname = model_path+"/model/"  # 模型参数保存目录
    height = width = 28

    # 加载测试数据
    imgs = [] #使用多种不同的预处理方法
    imgs_weight = [1,0.99,0.99] #不同预处理方法的结果权重
    try: #测试集图片
        imgs.append(utils.load_image(img_file,height,width))
    except:
	utils.warning(utils.get_trace())
	imgs.append([])
    try: #白纸手写照片
	if len(file_name.split('_')[1].split('.')[0])>=2 and int(file_name.split('_')[1][1:2]) > 0:
	    imgs_weight[1] = 5 
        imgs.append(utils.load_image(img_file, height, width, rotate=0, sobel=True, save_resize=True,ksize=5,dilate=1))
    except:
	utils.warning(utils.get_trace(),'infer')
	imgs.append([])
    try: #黑纸粗笔写照片
        imgs.append(utils.load_image(img_file, height, width, rotate=0, sobel=True, save_resize=True,ksize=3,dilate=6,erode=1))
    except:
	utils.warning(utils.get_trace(),'infer')
	imgs.append([])

    # 使用保存的模型参数+测试图片进行预测
    inferencer = Inferencer(
        # infer_func=softmax_regression, # uncomment for softmax regression
        # infer_func=multilayer_perceptron, # uncomment for MLP
        infer_func=utils.convolutional_neural_network,  # uncomment for LeNet5
        param_path=params_dirname,
        place=place)

    results = []
    results_sum = numpy.ndarray([])
    numpy.set_printoptions(precision=2)
    for i in xrange(len(imgs)):
        if len(imgs[i])==0:continue
        result = inferencer.infer({'img': imgs[i]}) #此输入img的各label概率
        result = numpy.where(result[0][0]>0.01 ,result[0][0],0) #概率<0.1%的直接设置为0
        print(result)
        results.append(result)
        print(numpy.argsort(result))
        results_sum = results_sum + result*imgs_weight[i]   #累加label下标概率
    #print(imgs_weight)
    #按概率加和排序
    lab = numpy.argsort(results_sum)  # probs and lab are the results of one batch data
    label = lab[-1]  #概率倒排最后一个
    weight = []
    for result in results:
        if numpy.argsort(result)[-1] == label:
            weight = result
    print("*label weight sort:")
    print(results_sum)
    print(lab)
    print("*img: %s" % img_file)
    print("*label: %d weight: %f" % (label,weight[label]))
    return 0,'',{'img':img_file, 'label':label, 'weight':list(weight.astype(str))}


if __name__ == '__main__':
    #default or cmd input img_file 
    img_file='./data/image/infer_62.jpeg'
    opts, args = getopt.getopt(sys.argv[1:], "p:", ["file_name="])
    if len(args) > 0 and len(args[0])>4:
        img_file = './data/image/' + args[0]

    ret = infer(img_file)
    print(ret)
