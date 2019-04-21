# -*- coding: UTF-8 -*-
from init_caffe import *
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import tempfile
from loss_functions import *
import numpy as np
import sys
sys.path.append('../lib/cython')
from connectivity import enforce_connectivity

from utils import *

trans_dim = 15

def normalize(bottom, dim):

    bottom_relu = L.ReLU(bottom)
    sum = L.Convolution(bottom_relu,
                        convolution_param = dict(num_output = 1, kernel_size = 1, stride = 1,
                                                 weight_filler = dict(type = 'constant', value = 1),
                                                 bias_filler = dict(type = 'constant', value = 0)),
                        param=[{'lr_mult':0, 'decay_mult':0}, {'lr_mult':0, 'decay_mult':0}])

    denom = L.Power(sum, power=(-1.0), shift=1e-12)
    denom = L.Tile(denom, axis=1, tiles=dim)

    return L.Eltwise(bottom_relu, denom, operation=P.Eltwise.PROD)


def conv_bn_relu_layer(bottom, num_out):
    conv1 = L.Convolution(bottom,
                          convolution_param=dict(num_output=num_out, kernel_size=3, stride=1, pad=1,
                                                 weight_filler=dict(type='gaussian', std=0.001),
                                                 bias_filler=dict(type='constant', value=0)),
                          # engine = P.Convolution.CUDNN),
                          param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}])
    bn1 = L.BatchNorm(conv1)
    bn1 = L.ReLU(bn1, in_place = True)

    return bn1

def conv_relu_layer(bottom, num_out):
    conv1 = L.Convolution(bottom,
                          convolution_param=dict(num_output=num_out, kernel_size=3, stride=1, pad=1,
                                                 weight_filler=dict(type='gaussian', std=0.001),
                                                 bias_filler=dict(type='constant', value=0)),
                          param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}])

    conv1 = L.ReLU(conv1, in_place = True)

    return conv1

# feature layer
def conv_relu_feature_layer(bottom, num_out):
    conv1 = L.Convolution(bottom,
                          convolution_param=dict(num_output=num_out, kernel_size=3, stride=1, pad=1,
                                                 weight_filler=dict(type='xavier', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)),
                          param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}])

    conv1 = L.ReLU(conv1, in_place = True)

    return conv1

def conv_relu_layer5(bottom, num_out):
    # 增加了孔算法,多了一个dilation参数
    conv1 = L.Convolution(bottom,
                          convolution_param=dict(num_output=num_out, kernel_size=3, stride=1, pad=2,dilation = 2,
                                                 weight_filler=dict(type='gaussian', std=0.001),
                                                 bias_filler=dict(type='constant', value=0)),
                          param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}])

    conv1 = L.ReLU(conv1, in_place = True)

    return conv1


def conv_down_relu_layer(bottom, num_out):

    conv1 = L.Convolution(bottom,
                          convolution_param=dict(num_output=num_out, kernel_size=3, stride=1, pad=1,
                                                 weight_filler=dict(type='xavier', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)),
                          # engine = P.Convolution.CUDNN),
                          param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}])

    conv1 = L.ReLU(conv1, in_place = True)

    return conv1

def conv_normalize_layer(bottom, num_out):

    conv1 = L.Convolution(bottom,
                          convolution_param=dict(num_output=num_out, kernel_size=1, stride=1, pad=0,
                                                 weight_filler=dict(type='xavier', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)),
                          # engine = P.Convolution.CUDNN),
                          param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}])

    norm = L.Normalize(conv1,
                       norm_param = dict(across_spatial = False,
                                       scale_filler = dict(type = 'constant', value = 20),
                                       channel_shared = False))
    return norm

def deconv_crop_layer(bottom, bottom2,num_out, size_kerbel,size_stride,num_offset):
    deconv1 = L.Deconvolution(bottom,
                              convolution_param = dict(num_output = num_out, kernel_size = size_kerbel, stride = size_stride, pad =0),
                                                       param = [{'lr_mult':0,'decay_mult':1},{'lr_mult':0, 'decay_mult':0}])
    feature_dsn = L.Crop(deconv1,bottom2,
                          crop_param = dict(axis = 2, offset = num_offset))

    return feature_dsn

def cnn_module(bottom, num_out):

    conv1_1 = conv_relu_layer(bottom, 64)
    conv1_2 = conv_relu_layer(conv1_1, 64)
    pool1 = L.Pooling(conv1_2, pooling_param = dict(kernel_size = 2, stride = 2, pad = 0, pool = P.Pooling.MAX))

    conv2_1 = conv_relu_layer(pool1, 128)
    conv2_2 = conv_relu_layer(conv2_1, 128)
    pool2 = L.Pooling(conv2_2, pooling_param = dict(kernel_size = 2, stride = 2, pad = 0, pool = P.Pooling.MAX))

    conv3_1 = conv_relu_layer(pool2, 256)
    conv3_2 = conv_relu_layer(conv3_1, 256)
    conv3_3 = conv_relu_layer(conv3_2, 256)
    pool3 = L.Pooling(conv3_3, pooling_param = dict(kernel_size= 2, stride = 2, pad= 0, pool=P.Pooling.MAX))

    conv4_1 = conv_relu_layer(pool3, 512)
    conv4_2 = conv_relu_layer(conv4_1, 512)
    conv4_3 = conv_relu_layer(conv4_2, 512)
    pool4 = L.Pooling(conv4_3, pooling_param=dict(kernel_size= 3, stride= 1, pad= 1, pool=P.Pooling.MAX))

    conv5_1 = conv_relu_layer5(pool4, 512)
    conv5_2 = conv_relu_layer5(conv5_1, 512)
    conv5_3 = conv_relu_layer5(conv5_2, 512)

    # conv1 side output
    conv1_2_down = conv_down_relu_layer(conv1_2,32)
    conv1_2_norm = conv_normalize_layer(conv1_2_down, 32)
    feature_dsn1 = L.Crop(conv1_2_norm, bottom)

    # conv2 side output
    conv2_2_down = conv_down_relu_layer(conv2_2, 64)
    conv2_2_norm = conv_normalize_layer(conv2_2_down, 64)
    feature_dsn2 = deconv_crop_layer(conv2_2_norm,bottom,64,4, 2, 1)

    # conv3 side output
    conv3_3_down = conv_down_relu_layer(conv3_3, 128)
    conv3_3_norm = conv_normalize_layer(conv3_3_down, 64)
    feature_dsn3 = deconv_crop_layer(conv3_3_norm, bottom, 64, 8, 4, 2)

    # conv4 side output
    conv4_3_down = conv_down_relu_layer(conv4_3, 256)
    conv4_3_norm = conv_normalize_layer(conv4_3_down, 128)
    feature_dsn4 = deconv_crop_layer(conv4_3_norm, bottom, 128, 16, 8, 4)

    # conv5 side output
    conv5_3_down = conv_down_relu_layer(conv5_3, 256)
    conv5_3_norm = conv_normalize_layer(conv5_3_down, 128)
    feature_dsn5 = deconv_crop_layer(conv5_3_norm, bottom, 128, 16, 8, 4)

    # concat multiscale feature layer
    feature = L.Concat(feature_dsn1, feature_dsn2, feature_dsn3, feature_dsn4, feature_dsn5,
                       concat_param = dict(axis = 1))



    # the feature layer of del

    conv_dim = conv_relu_feature_layer(feature, 256)

    conv_dsp = L.Convolution(conv_dim,
                    convolution_param=dict(num_output= 64, kernel_size=1, stride=1, pad=0,
                                            weight_filler=dict(type='xavier', std=0.01),
                                            bias_filler=dict(type='constant', value=0)),
                    param=[{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}])



    # the layer of ssn
    conv_comb = conv_relu_layer(feature,num_out)
    # conv_dsp == Feature Embedding Space
    return conv_comb, conv_dsp


# 得到像素-超像素联系
def compute_assignments(spixel_feat, pixel_features,
                        spixel_init, num_spixels_h,
                        num_spixels_w, num_spixels, num_channels):

    num_channels = int(num_channels)
    # Passoc 作者自己写的一个层，暂时还没看懂
    pixel_spixel_neg_dist = L.Passoc(pixel_features, spixel_feat, spixel_init,
                                     spixel_feature2_param =\
          dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w, scale_value = -1.0))

    # Softmax to get pixel-superpixel relative soft-associations
    # Softmax获得像素超像素相对软关联
    pixel_spixel_assoc = L.Softmax(pixel_spixel_neg_dist)

    return pixel_spixel_assoc

# 计算最后的超像素与像素的联系
def compute_final_spixel_labels(pixel_spixel_assoc,
                                spixel_init,
                                num_spixels_h, num_spixels_w):

    # Compute new spixel indices
    # 计算新的像素指数
    rel_label = L.ArgMax(pixel_spixel_assoc, argmax_param = dict(axis = 1),
                         propagate_down = False)
    new_spix_indices = L.RelToAbsIndex(rel_label, spixel_init,
                                       rel_to_abs_index_param = dict(num_spixels_h = int(num_spixels_h),
                                                                     num_spixels_w = int(num_spixels_w)),
                                                                     propagate_down = [False, False])

    return new_spix_indices


def decode_features(pixel_spixel_assoc, spixel_feat, spixel_init,
                    num_spixels_h, num_spixels_w, num_spixels, num_channels):

    num_channels = int(num_channels)

    # Reshape superpixel features to k_h x k_w
    spixel_feat_reshaped = L.Reshape(spixel_feat,
                                      reshape_param = dict(shape = {'dim':[0,0,num_spixels_h,num_spixels_w]}))

    # Concatenate neighboring superixel features
    concat_spixel_feat = L.Convolution(spixel_feat_reshaped,
                                        name = 'concat_spixel_feat_' + str(num_channels),
                                        convolution_param = dict(num_output = num_channels * 9,
                                                                 kernel_size = 3,
                                                                 stride = 1,
                                                                 pad = 1,
                                                                 group = num_channels,
                                                                 bias_term = False),
                                                                 param=[{'name': 'concat_spixel_feat_' + str(num_channels),
                                                                        'lr_mult':0, 'decay_mult':0}])

    # Spread features to pixels
    flat_concat_label = L.Reshape(concat_spixel_feat,
                                  reshape_param = dict(shape = {'dim':[0, 0, 1, num_spixels]}))
    img_concat_spixel_feat = L.Smear(flat_concat_label, spixel_init)

    tiled_assoc = L.Tile(pixel_spixel_assoc,
                         tile_param = dict(tiles = num_channels))

    weighted_spixel_feat = L.Eltwise(img_concat_spixel_feat, tiled_assoc,
                                     eltwise_param = dict(operation = P.Eltwise.PROD))
    recon_feat = L.Convolution(weighted_spixel_feat,
                               name = 'recon_feat_' + str(num_channels),
                               convolution_param = dict(num_output = num_channels,
                                                        kernel_size = 1,
                                                        stride = 1,
                                                        pad = 0,
                                                        group = num_channels,
                                                        bias_term = False,
                                                        weight_filler = dict(type = 'constant', value = 1.0)),
                                                        param=[{'name': 'recon_feat_' + str(num_channels),
                                                               'lr_mult':0, 'decay_mult':0}])

    return recon_feat


# 建立像素和超像素之间的联系
def exec_iter(spixel_feat, trans_features, spixel_init,
              num_spixels_h, num_spixels_w, num_spixels,
              trans_dim):

    # Compute pixel-superpixel assignments
    #  计算像素超像素分配
    pixel_assoc = \
        compute_assignments(spixel_feat, trans_features,
                            spixel_init, num_spixels_h,
                            num_spixels_w, num_spixels, trans_dim)
    # Compute superpixel features from pixel assignments
    # 通过像素分配计算超像素特征
    spixel_feat1 = L.SpixelFeature2(trans_features,
                                    pixel_assoc,
                                    spixel_init,
                                    spixel_feature2_param =\
        dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w))

    return spixel_feat1

# 创建ssn网络
def create_ssn_net(img_height, img_width,
                   num_spixels, pos_scale, color_scale,
                   num_spixels_h, num_spixels_w, num_steps,
                   phase = None):

    n = caffe.NetSpec()

    if phase == 'TRAIN':
        n.img, n.spixel_init, n.feat_spixel_init, n.label, n.problabel, n.seg_label = \
            L.Python(python_param = dict(module = "input_patch_data_layer", layer = "InputRead", param_str = "TRAIN_1000000_" + str(num_spixels)),
                     include = dict(phase = 0),
                     ntop = 6)

    elif phase == 'TEST':
        n.img, n.spixel_init, n.feat_spixel_init, n.label, n.problabel, n.seg_label= \
            L.Python(python_param = dict(module = "input_patch_data_layer", layer = "InputRead", param_str = "VAL_10_" + str(num_spixels)),
                     include = dict(phase = 1),
                     ntop = 6)
    else:
        # im:10  ——表示对待识别样本进行数据增广的数量，该值的大小可自行定义。但一般会进行5次crop，将整幅图像分为多个flip。该值为10则表示会将待识别的样本分为10部分输入到网络进行识别。
        # 如果相对整幅图像进行识别而不进行图像数据增广，则可将该值设置为1.
        # dim:3 ——该值表示处理的图像的通道数，若图像为RGB图像则通道数为3，设置该值为3；若图像为灰度图，通道数为1则设置该值为1.
        # dim:32 ——图像的长度，可以通过网络配置文件中的数据层中的crop_size来获取。
        # dim:32——图像的宽度，可以通过网络配置文件中的数据层中的crop_size来获取。
        n.img = L.Input(shape=[dict(dim=[1, 3, img_height, img_width])])
        n.spixel_init = L.Input(shape=[dict(dim=[1, 1, img_height, img_width])])
        n.feat_spixel_init = L.Input(shape=[dict(dim=[1, 1, img_height, img_width])])
        n.bound_param = L.Input(shape=[dict(dim=[1, 1, 1, 1])])
        n.minsize_param = L.Input(shape=[dict(dim=[1, 1, 1, 1])])

    # 我也不知道这里怎么得出pixel_features
    # lib/video_prop_networks/lib/caffe/src/caffe/layers
    n.pixel_features = L.PixelFeature(n.img,
                                      pixel_feature_param = dict(type = P.PixelFeature.POSITION_AND_RGB,
                                                                 pos_scale = float(pos_scale),
                                                                 color_scale = float(color_scale)))

    ### Transform Pixel features trans_dim = 15
    n.trans_features, n.conv_dsp = cnn_module(n.pixel_features, trans_dim)

    # Initial Superpixels
    n.init_spixel_feat = L.SpixelFeature(n.trans_features, n.feat_spixel_init,
                                         spixel_feature_param =\
        dict(type = P.SpixelFeature.AVGRGB, rgb_scale = 1.0, ignore_idx_value = -10,
             ignore_feature_value = 255, max_spixels = int(num_spixels)))

    ### Iteration-1
    n.spixel_feat1 = exec_iter(n.init_spixel_feat, n.trans_features,
                               n.spixel_init, num_spixels_h,
                               num_spixels_w, num_spixels, trans_dim)

    ### Iteration-2
    n.spixel_feat2 = exec_iter(n.spixel_feat1, n.trans_features,
                               n.spixel_init, num_spixels_h,
                               num_spixels_w, num_spixels, trans_dim)

    ### Iteration-3
    n.spixel_feat3 = exec_iter(n.spixel_feat2, n.trans_features,
                               n.spixel_init, num_spixels_h,
                               num_spixels_w, num_spixels, trans_dim)

    ### Iteration-4
    n.spixel_feat4 = exec_iter(n.spixel_feat3, n.trans_features,
                               n.spixel_init, num_spixels_h,
                               num_spixels_w, num_spixels, trans_dim)

    if num_steps == 5:
        ### Iteration-5
        n.final_pixel_assoc  = \
            compute_assignments(n.spixel_feat4, n.trans_features,
                                n.spixel_init, num_spixels_h,
                                num_spixels_w, num_spixels, trans_dim)

    elif num_steps == 10:
        ### Iteration-5
        n.spixel_feat5 = exec_iter(n.spixel_feat4, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-6
        n.spixel_feat6 = exec_iter(n.spixel_feat5, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-7
        n.spixel_feat7 = exec_iter(n.spixel_feat6, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-8
        n.spixel_feat8 = exec_iter(n.spixel_feat7, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-9
        n.spixel_feat9 = exec_iter(n.spixel_feat8, n.trans_features,
                                   n.spixel_init, num_spixels_h,
                                   num_spixels_w, num_spixels, trans_dim)

        ### Iteration-10
        # 得到超像素与像素之间的软链接
        n.final_pixel_assoc  = \
            compute_assignments(n.spixel_feat9, n.trans_features,
                                n.spixel_init, num_spixels_h,
                                num_spixels_w, num_spixels, trans_dim)


    if phase == 'TRAIN' or phase == 'TEST':

        # Compute final spixel features
        # 紧凑型损失
        n.new_spixel_feat = L.SpixelFeature2(n.pixel_features,
                                             n.final_pixel_assoc,
                                             n.spixel_init,
                                             spixel_feature2_param =\
            dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w))

        # 得到最后的超像素标签
        #计算最后的超像素与像素的联系
        n.new_spix_indices = compute_final_spixel_labels(n.final_pixel_assoc,
                                                         n.spixel_init,
                                                         num_spixels_h, num_spixels_w)

        n.recon_feat2 = L.Smear(n.new_spixel_feat, n.new_spix_indices,
                                propagate_down = [True, False])
        n.loss1, n.loss2 = position_color_loss(n.recon_feat2, n.pixel_features,
                                               pos_weight = 0.001,
                                               col_weight = 0.0)


        # Convert pixel labels to spixel labels
        # 任务特征重建损失
        # 将像素标签转化为超像素标签（这里应该是硬链接，用来计算损失函数）
        # 个人感觉spixel_label和上面的
        n.spixel_label = L.SpixelFeature2(n.problabel,
                                          n.final_pixel_assoc,
                                          n.spixel_init,
                                          spixel_feature2_param =\
            dict(num_spixels_h = num_spixels_h, num_spixels_w = num_spixels_w))
        # Convert spixel labels back to pixel labels
        # 将超像素标签转回到像素标签
        n.recon_label = decode_features(n.final_pixel_assoc, n.spixel_label, n.spixel_init,
                                        num_spixels_h, num_spixels_w, num_spixels, num_channels = 50)

        # superpixel_pooling
        n.superpixel_pooling_out, n.superpixel_seg_label = L.SuperpixelPooling(n.conv_dsp, n.seg_label,
                                                                               n.new_spix_indices,
                                                                               superpixel_pooling_param=dict(
                                                                                   pool_type=P.Pooling.AVE), ntop=2)
        n.recon_label = L.ReLU(n.recon_label, in_place = True)
        n.recon_label2 = L.Power(n.recon_label, power_param = dict(shift = 1e-10))
        n.recon_label3 = normalize(n.recon_label2, 50)
        n.loss3 = L.LossWithoutSoftmax(n.recon_label3, n.label,
                                       loss_param = dict(ignore_label = 255),
                                       loss_weight = 1.0)

        # 这里需要获得超像素



        # the loss of del
        n.sim_loss = L.SimilarityLoss(n.superpixel_pooling_out,n.superpixel_seg_label, n.new_spix_indices,
                                      loss_weight = 0, similarity_loss_param = dict(sample_points = 1))



    else:
        n.new_spix_indices = compute_final_spixel_labels(n.final_pixel_assoc,
                                                         n.spixel_init,
                                                         num_spixels_h, num_spixels_w)
        n.segmentation = L.EgbSegment(n.conv_dsp, n.new_spix_indices, n.bound_param, n.minsize_param,
                                      egb_segment_param=dict(bound=3, min_size=10))
#  NetSpec 是包含Tops（可以直接赋值作为属性）的集合。调用 NetSpec.to_proto 创建包含所有层(layers)的网络参数，这些层(layers)需要被赋值，并使用被赋值的名字。
    return n.to_proto()


def load_ssn_net(img_height, img_width,
                 num_spixels, pos_scale, color_scale,
                 num_spixels_h, num_spixels_w, num_steps):

    net_proto = create_ssn_net(img_height, img_width,
                               num_spixels, pos_scale, color_scale,
                               num_spixels_h, num_spixels_w, int(num_steps))

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()

    return caffe.Net(f.name, caffe.TEST)

# train Net
def get_ssn_net(img_height, img_width,
                num_spixels, pos_scale, color_scale,
                num_spixels_h, num_spixels_w, num_steps,
                phase):

    # Create the prototxt
    net_proto = create_ssn_net(img_height, img_width,
                               num_spixels, pos_scale, color_scale,
                               num_spixels_h, num_spixels_w, int(num_steps), phase)

    # Save to temporary file and load
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()
    return f.name
