# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from config import *
sys.path.insert(0, CAFFEDIR + 'python')
import caffe
import numpy as np
from math import ceil



#set device
caffe.set_device(0)
caffe.set_mode_gpu()   
# multi-GPU
#gpus = [0, 1]
#caffe.set_device(gpus)
        
solver = caffe.AdamSolver('solver.prototxt')
  
# 等价于solver文件中的max_iter，即最大解算次数  
niter = 600000
# 每隔100次收集一次数据  
display= 100
  
# 每次测试进行100次解算，10000/100  
test_iter = 200
# 每500次训练进行一次测试（100次解算）， 
test_interval = 10000
  
#初始化 
train_loss = np.zeros(int(ceil(niter * 1 / display)))   
test_loss = np.zeros(int(ceil(niter * 1 / test_interval)))  
 
# iteration 0，不计入  
solver.step(1)  
  
# 辅助变量  
_train_loss = 0; _test_loss = 0
# 进行解算  
for it in range(niter):  
    # 进行一次解算  
    solver.step(1)  
    # 每迭代一次，训练batch_size张图片  
    _train_loss += (solver.net.blobs['loss0'].data * 0.1 + solver.net.blobs['loss1'].data * 0.00001 + solver.net.blobs['loss3'].data *1)
    if it % display == 0:  
        # 计算平均train loss  
        train_loss[it // display] = _train_loss / display
        _train_loss = 0  
  
    if it % test_interval == 0:  
        for test_it in range(test_iter):  
            # 进行一次测试  
            solver.test_nets[0].forward()  
            # 计算test loss  
            _test_loss += (solver.test_nets[0].blobs['loss0'].data * 0.1 + solver.test_nets[0].blobs['loss1'].data*0.00001 + solver.test_nets[0].blobs['loss3'].data*1)
        # 计算平均test loss  
        test_loss[it / test_interval] = _test_loss / test_iter  
        # 计算平均test accuracy  
        
        _test_loss = 0  
    
    if it % 10000 == 0:
        # train loss -> 绿色  
        l1 = plt.plot(display * np.arange(len(train_loss)), train_loss, 'g')
        # test loss -> 黄色  
        l2 = plt.plot(test_interval * np.arange(len(test_loss)), test_loss, 'y')
        # test accuracy -> 红色  
        #ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')  
        plt.legend(fontsize=10)  
        plt.xlabel('iteration')  
        plt.ylabel('loss')  
        #ax2.set_ylabel('accuracy')  
        plt.savefig('loss_' + str(it) + '_.jpg') 
        
# 绘制train loss、test loss和accuracy曲线  
# print '\nplot the train loss and test accuracy\n'
#_, ax1 = plt.subplots()  
#ax2 = ax1.twinx()  
  

