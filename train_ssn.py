# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

import numpy as np
import sys
import argparse

from init_caffe import *
from config import *
from create_solver import *
from create_net import get_ssn_net
from utils import get_spixel_init, initialize_net_weight

def train_net(l_rate, num_steps, caffe_model = None):

    # Solver Params
    lr = float(l_rate)
    prefix = './models/intermediate_bsds_model_' + str(l_rate) + '_'
    test_iter = 200
    iter_size = 1
    test_interval = 10000
    # num_iter = 1000000
    num_iter = 10000
    # snapshot_iter = 10000
    snapshot_iter = 100
    debug_info = False

    # Net params
    patch_height = TRAIN_PATCH_WIDTH
    patch_width = TRAIN_PATCH_HEIGHT
    num_spixels = 2300
    spixel_initmap, feat_spixel_initmap, num_spixels_w, num_spixels_h = \
        get_spixel_init(num_spixels,
                        patch_width,
                        patch_height)

    pos_scale = 0.25
    color_scale = 0.26

    train_net_file = get_ssn_net(patch_height, patch_width, int(num_spixels_w * num_spixels_h),
                                 float(pos_scale), float(color_scale),
                                 num_spixels_h, num_spixels_w, num_steps, phase = 'TRAIN')
    test_net_file = get_ssn_net(patch_height, patch_width, int(num_spixels_w * num_spixels_h),
                                float(pos_scale), float(color_scale),
                                num_spixels_h, num_spixels_w, num_steps, phase = 'TEST')

    solver_proto = create_solver_proto(train_net_file,
                                       test_net_file,
                                       lr,
                                       prefix,
                                       test_iter = test_iter,
                                       test_interval = test_interval,
                                       max_iter=num_iter,
                                       iter_size=iter_size,
                                       snapshot=snapshot_iter,
                                       display = 1,
                                       debug_info=debug_info)
    solver = create_solver(solver_proto)

    if caffe_model is None:
        initialize_net_weight(solver.net)
    else:
        solver.net.copy_from(caffe_model)

    solver.solve()

    # 画图

    # 最大解算次数
    niter = 10000

    # 每隔20次收集一次数据
    display = 20
    # 初始化
    train_loss = np.zeros(ceil(niter * 1.0 / display))

    # iteration 0，不计入
    solver.step(1)

    # 辅助变量
    _train_loss = 0
    # 进行解算
    for it in range(niter):
        # 进行一次解算
        solver.step(1)
        # 每迭代一次，训练batch_size张图片
        _train_loss += solver.net.blobs['loss0'].data
        if it % display == 0:
            # 计算平均train loss
            train_loss[it // display] = _train_loss / display
            _train_loss = 0

     # 绘制train loss、test loss和accuracy曲线
    print '\nplot the train loss and test accuracy\n'
    _, ax1 = plt.subplots()

    # train loss -> 绿色
    ax1.plot(display * arange(len(train_loss)), train_loss, 'g')

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--l_rate', type=float, default=0.0001)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--caffemodel', type=str, default=None)

    var_args = parser.parse_args()
    train_net(var_args.l_rate,
              var_args.num_steps,
              var_args.caffemodel)

if __name__ == '__main__':
    main()
