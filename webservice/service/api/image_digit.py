#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: image_digit.py
Desc: 手写数字识别
Author: yanjingang(yanjingang@baidu.com)
Date: 2018/12/19 16:08
"""

import sys
import json
import tornado.web
import tornado.ioloop
digit_path = '../../sample/recognize_digits/train/'
sys.path.append(digit_path)
import utils
import infer

class ApiImageDigit(tornado.web.RequestHandler):
    """API逻辑封装"""

    def get(self):
        """get请求处理"""
        try:
            result = self.execute()
        except:
            utils.error('execute fail ' + utils.get_trace(), ApiImageDigit)
            result = {'code': 1, 'msg': '查询失败'}
        utils.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                      + str(result['code']) + '][' + str(result['msg']) + '][' + str(result['data']) + ']', ApiImageDigit)
        self.write(json.dumps(result))

    def post(self):
        """post请求处理"""
        try:
            result = self.execute()
        except:
            utils.error('execute fail ' + utils.get_trace(), ApiImageDigit)
            result = {'code': 1, 'msg': '查询失败'}
        utils.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                      + str(result['code']) + '][' + str(result['msg']) + ']', ApiImageDigit)
        self.write(json.dumps(result))

    def execute(self):
        """执行业务逻辑"""
        utils.info('API REQUEST INFO[' + self.request.path + '][' + self.request.method + ']['
                      + self.request.remote_ip + '][' + str(self.request.arguments) + ']', ApiImageDigit)
        img_file = self.get_argument('img_file', '')
        model_path = self.get_argument('model_path', '')
        if img_file == '':
            return {'code': 2, 'msg': 'img_file不能为空'}
        if model_path == '':
           model_path = digit_path 
        res = {}

        try:
            ret, msg, res = infer.main(img_file,model_path)
            if ret != 0:
                utils.error('execute fail [' + img_file + '] ' + msg, ApiImageDigit)
                return {'code': 4, 'msg': '查询失败'}
        except:
            utils.error('execute fail [' + img_file + '] ' + utils.get_trace(), ApiImageDigit)
            return {'code': 5, 'msg': '查询失败'}

        # 组织返回格式
        return {'code': 0, 'msg': 'success', 'data': res}
