#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: api_digit.py
Desc: 识别手写数字 API
Demo: 
    cd /home/work/paddle/piglab/service/ && nohup python api_paddle.py > log/api_paddle.log &
    http://www.yanjingang.com:8020/piglab/image/digit?img_file=/home/work/paddle/sample/recognize_digits/train/data/image/infer_62.jpeg
    http://yq01-aladdin-gpu1.yq01.baidu.com:8020/piglab/image/digit?img_file=/home/aladdin/yanjingang/paddle/sample/recognize_digits/train/data/tmp/infer_62_dilation.jpeg

    ps aux | grep api_paddle.py |grep -v grep| cut -c 9-15 | xargs kill -9
Author: yanjingang(yanjingang@baidu.com)
Date: 2018/12/19 19:39
"""

import sys
import json
import tornado.ioloop
import tornado.web
import tornado.httpserver

sys.path.append('./api/')
from image_digit import ApiImageDigit


if __name__ == '__main__':
    """服务入口"""
    port = 8020

    # 路由
    app = tornado.web.Application(
        handlers=[(r'/piglab/image/digit', ApiImageDigit)]
    )

    # 启动服务
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

