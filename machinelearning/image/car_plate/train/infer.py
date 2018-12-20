#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
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
import utils

def main():
    # 定义训练和测试数据batch reader
    mnist_path = '/home/work/.cache/paddle/dataset/mnist/'
    test_image    = mnist_path + 't10k-images-idx3-ubyte.gz'
    test_label    = mnist_path + 't10k-labels-idx1-ubyte.gz'
    test_reader = paddle.batch(
	#paddle.dataset.mnist.test(),
	utils.reader_creator(test_image,test_label,buffer_size=100), #自己读取mnist测试集 
	batch_size=64)

    # 是否使用GPU
    use_cuda = False  # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    # 模型参数保存目录
    params_dirname = "model/"

    # 加载测试数据
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = utils.load_image(cur_dir + '/data/image/infer_3.png')

    # 使用保存的模型参数+测试图片进行预测
    inferencer = Inferencer(
        # infer_func=softmax_regression, # uncomment for softmax regression
        # infer_func=multilayer_perceptron, # uncomment for MLP
        infer_func=utils.convolutional_neural_network,  # uncomment for LeNet5
        param_path=params_dirname,
        place=place)
    results = inferencer.infer({'img': img})
    lab = numpy.argsort(results)  # probs and lab are the results of one batch data
    print("Inference result of data/image/infer_3.png is: %d" % lab[0][0][-1])


if __name__ == '__main__':
    main()
