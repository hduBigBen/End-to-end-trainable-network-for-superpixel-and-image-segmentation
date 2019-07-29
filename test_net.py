#!/usr/bin/env python
# coding: utf-8
"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Author: Varun Jampani
"""

import numpy as np
import scipy.io as sio
import os
import scipy
from scipy.misc import fromimage
from scipy.misc import imsave
from PIL import Image
import argparse
import cv2

from init_caffe import *
from config import *
from utils import *
from fetch_and_transform_data import fetch_and_transform_data, transform_and_get_spixel_init
from create_net import load_ssn_net

import sys
sys.path.append('../lib/cython')
from connectivity import enforce_connectivity

def compute_spixels(data_type, n_spixels, num_steps,
                    caffe_model, out_folder, is_connected = True):

    image_list = IMG_LIST[data_type]

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    p_scale = 0.40
    color_scale = 0.26

    bound = range(1, 71, 2)
    # bound = range(31, 71, 2)
    for i in range(len(bound)):
        threshold = bound[i] / 100.0
        if threshold <= 0.3:
            min_size = [0, 1, 2, 3]
        else:
            # min_size = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            min_size = [0, 1, 2, 3, 4, 5, 6]

        for j in range(len(min_size)):
            new_save_root = os.path.join(out_folder, 'threshold_{:.2f}_{}'.format(threshold, min_size[j]))
            if not os.path.exists(new_save_root):
                os.mkdir(new_save_root)

            with open(image_list) as list_f:
                for imgname in list_f:
                    print(imgname)
                    imgname = imgname[:-1]
                    [inputs, height, width] = \
                        fetch_and_transform_data(imgname, data_type,
                                                 ['img', 'label', 'problabel'],
                                                 int(n_spixels))

                    height = inputs['img'].shape[2]
                    width = inputs['img'].shape[3]
                    [spixel_initmap, feat_spixel_initmap, num_spixels_h, num_spixels_w] = \
                        transform_and_get_spixel_init(int(n_spixels), [height, width])

                    dinputs = {}
                    dinputs['img'] = inputs['img']
                    dinputs['spixel_init'] = spixel_initmap
                    dinputs['feat_spixel_init'] = feat_spixel_initmap

                    # 新加
                    thresholda = np.random.randint(bound[i], bound[i]+1, size=(1, 1, 1, 1))
                    thresholda = thresholda/100.0
                    min_sizea = np.random.randint(min_size[j], min_size[j]+1, size=(1, 1, 1, 1))
                    dinputs['bound_param'] = thresholda
                    dinputs['minsize_param'] = min_sizea


                    pos_scale_w = (1.0 * num_spixels_w) / (float(p_scale) * width)
                    pos_scale_h = (1.0 * num_spixels_h) / (float(p_scale) * height)
                    pos_scale = np.max([pos_scale_h, pos_scale_w])

                    net = load_ssn_net(height, width, int(num_spixels_w * num_spixels_h),
                                       float(pos_scale), float(color_scale),
                                       num_spixels_h, num_spixels_w, int(num_steps))

                    if caffe_model is not None:
                        net.copy_from(caffe_model)
                    else:
                        net = initialize_net_weight(net)

                    num_spixels = int(num_spixels_w * num_spixels_h)
                    result = net.forward_all(**dinputs)




                    given_img = fromimage(Image.open(IMG_FOLDER[data_type] + imgname + '.jpg'))

                    # 保存mat格式
                    # out2 = net.blobs['segmentation'].data[0].copy()
                    out2 = net.blobs['segmentation'].data[0].copy().astype(int)
                    # print out2.shape

                    # out2 = np.squeeze(net.blobs['segmentation'].data).astype(int)
                    # print out2.shape
                    #

                    if enforce_connectivity:
                        segment_size = (given_img.shape[0] * given_img.shape[1]) / (int(n_spixels) * 1.0)
                        min_size1 = int(0.24 * segment_size)
                        max_size1 = int(100 * segment_size)
                        out2 = enforce_connectivity(out2, min_size1, max_size1)[0]
                    # kernel = np.ones((3,3),np.uint16)

                    # out2 = out2.astype(dtype=np.uint16)
                    # print "change the dim"
                    # out2 = out2[None, :, :]
                    # print out2.shape
                    # out2 = out2.transpose((1, 2, 0)).astype(dtype=np.uint16)

                    # open
                    # out2 = cv2.morphologyEx(out2, cv2.MORPH_OPEN, kernel)
                    out2 = out2[None, :, :]
                    out2 = out2.transpose((1, 2, 0)).astype(dtype=np.uint16)

                    # out2 = out2.astype(dtype=np.uint16)


                    sio.savemat(new_save_root + '/' + imgname + '.mat', {'Segmentation': out2})

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datatype', type=str, required=True)
    parser.add_argument('--n_spixels', type=int, required=True)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--caffemodel', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--is_connected', type=bool, default=True)

    var_args = parser.parse_args()
    compute_spixels(var_args.datatype,
                    var_args.n_spixels,
                    var_args.num_steps,
                    var_args.caffemodel,
                    var_args.result_dir,
                    var_args.is_connected)

if __name__ == '__main__':
    main()
