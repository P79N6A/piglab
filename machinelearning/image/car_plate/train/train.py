#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 车牌识别模型训练
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
    img_path = '/home/work/paddle/sample/car_platenumber/train/data/'
    train_images   = img_path + 'images_train/'
    test_images    = img_path + 'images_test/'
    label_idx = utils.image_label_dict(img_path)  #label词典
    train_reader = paddle.batch(paddle.reader.shuffle(
	#utils.image_reader_creator(train_images,60,180,label_idx), #训练集 reader
	utils.image_reader_creator(train_images,28,28,label_idx), #训练集 reader
	buf_size=500),
        batch_size=64)
    test_reader = paddle.batch(
	#utils.image_reader_creator(test_images,60,180,label_idx), #测试集reader
	utils.image_reader_creator(test_images,28,28,label_idx), #测试集reader
	batch_size=64)

    # 是否使用GPU
    use_cuda = False  # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    # 创建训练器(train_func损失函数; place是否使用gpu; optimizer_func优化器)
    trainer = Trainer(
        train_func=utils.train_program, place=place, optimizer_func=utils.optimizer_program)

    # 模型参数保存目录
    params_dirname = "model/"
    # 定义event_handler，输出训练过程中的结果
    lists = []
    def event_handler(event):
        if isinstance(event, EndStepEvent):  # 每步触发事件
            if event.step % 100 == 0:
                # event.metrics maps with train program return arguments.
                # event.metrics[0] will yeild avg_cost and event.metrics[1] will yeild acc in this example.
                print("Pass %d, Batch %d, Cost %f" % (event.step, event.epoch,
                                                      event.metrics[0]))
        if isinstance(event, EndEpochEvent): # 每次迭代触发事件
            # test的返回值就是train_func的返回值
            avg_cost, acc = trainer.test(
                reader=test_reader, feed_order=['img', 'label'])
            print("Test with Epoch %d, avg_cost: %s, acc: %s" %
                  (event.epoch, avg_cost, acc))
            # 保存模型参数
            trainer.save_params(params_dirname)
            # 保存训练结果损失情况
            lists.append((event.epoch, avg_cost, acc))

    # 开始训练模型
    trainer.train(
        num_epochs=5,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=['img', 'label'])

    # 找到训练误差最小的一次结果(找完没用？trainer.save_params()自动做了最优选择？)
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
    print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))

    # 加载测试数据
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = utils.load_image(cur_dir + '/data/image/infer_3.png')

    # 使用保存的模型参数+测试图片进行预测
    inferencer = Inferencer(
        # infer_func=utils.softmax_regression, # uncomment for softmax regression
        # infer_func=utils.multilayer_perceptron, # uncomment for MLP
        infer_func=utils.convolutional_neural_network,  # uncomment for LeNet5
        param_path=params_dirname,
        place=place)
    results = inferencer.infer({'img': img})
    lab = numpy.argsort(results)  # probs and lab are the results of one batch data
    print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])


if __name__ == '__main__':
    main()
