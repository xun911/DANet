# 开发时间 2022/3/17 10:50
from typing import Sequence
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np
# from torch import batch_norm
import dataloader
import argparse
from sklearn.model_selection import train_test_split
import NetModel
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD,Adam
import matplotlib
import tensorflow as tf
from tqdm import tqdm
from utils import getGpuMrmory,save_excel
import matplotlib
# matplotlib.use('module://backend_interagg')

import matplotlib.pyplot as plt
import pickle
import os
import time
from Common import CheckFolder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau,ModelCheckpoint
#选择GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#linux 报错可用：cuDNN launch failure : input shape
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

'''
计算CapgMyo DB-a 18个人的训练准确率平均值
'''

#------------------------------------------------------#
#数据加载
#------------------------------------------------------#

#------------------------------------------------------#
#定义model
#------------------------------------------------------#

def modelFit_TF(savePath):
    #学习率回调
    # reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    checkpoint = ModelCheckpoint(filepath=savePath, monitor='val_accuracy', mode='auto', save_best_only=True,save_weights_only=False)
    # 模型训练计时
    start = time.time()

    #模型训练l
    trainCount=dataloader.getDataDictCount('data/pretrain/'+dataset+'/trainCount.npy')
    testCount = dataloader.getDataDictCount('data/pretrain/'+dataset+'/testCount.npy')

    tfrecord_train =dataloader.GetTfFilesList('train',dataset)    #  'data/pretrain/trainall.tfrecords'
    tfrecord_test = dataloader.GetTfFilesList('test',dataset)
    dataset_train = dataloader.tfrecords_reader_dataset(tfrecord_train,trainCount,batch_size,dataset,epoch)
    dataset_test = dataloader.tfrecords_reader_dataset(tfrecord_test,testCount,batch_size,dataset,epoch)
    # dataset_test = dataloader.tfrecords_reader_dataset(tfrecord_test)
    if (dataset=="dba")|(dataset=="dbb-1")|(dataset=="dbb-all")|(dataset=="dbb-all-session"):
        input_shape=[150,128]
    elif (dataset=="femg"):
        input_shape=[150,64]
    elif dataset=='bio':
        input_shape=[150,6]
    else:
        input_shape = [9, 1, 800]
    model = NetModel.model_2SRNN(input_shape=input_shape, classes=classes)
    model.fit(dataset_train,
              # batch_size=256,
              # validation_data=dataset_test,
              steps_per_epoch=(trainCount// batch_size)+1,
                        epochs=epoch,
              validation_steps=(testCount// batch_size)+1,
              # max_queue_size=30,
             validation_data=dataset_test,

              verbose=2,
              # shuffle=True,
              use_multiprocessing=False,
              # workers=8,
              # validation_data=get_train_batch(valX1, valX2,
              #                                 valX3, valY1, batch_size),
              callbacks=[checkpoint]
              )
    preds_test = model.evaluate(dataset_test,steps=(testCount// batch_size)+1)
    print("Test Loss = " + str(preds_test[0]))
    print("Test Accuracy = " + str(preds_test[1]))
    end = time.time()



#根据epoch动态改变学习率
def lr_schedule(epoch):
    lr = 1e-1
    if (epoch >= 16)&(epoch<24):
        lr = 1e-2
    elif epoch >= 24:
        lr = 1e-3
    print('Learning rate: ', lr)
    return lr


# predictions = model.predict(x = valX, batch_size=batch_size)
#print(classification_report(valY.argmax(axis=1.h5), predictions.argmax(axis=1.h5), target_names=lb.classes_))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument('-d', '--dataset', required=True,
    #                 help='path of input dataset of mat')
    # ap.add_argument('-s', '--subject', required=False,
    #                 help='if not pretrained, subject is required, like 001')
    ap.add_argument('-d', '--dataset', default='bio', choices=['dba', 'dbb-all', 'dbb-all-session', 'femg', 'bio'],
                    help='select ninpro data')
    # ap.add_argument('-d', '--dataset', default='data/capgMyo/DBa',
    #                 help='path of input dataset of mat')
    ap.add_argument('-s', '--subject', default='001',
                    help='if not pretrained, subject is required, like 001')
    ap.add_argument('-p', '--pretrained', action='store_true',
                    help='if pretrained give the parser --pretrained, else not')
    ap.add_argument('-m', '--pretrained model', default=True,
                    help='if use pretrained model, should give path of pretrained model')
    ap.add_argument('-b', '--batch_size', default=520,
                    help='batch size')
    ap.add_argument('-e', '--epoch', default=100,
                    help='epoch')
    ap.add_argument('-pl', '--plot', default='plot/1.png',
                    help='path to save accuracy/loss plot')
    ap.add_argument('-sm', '--model', default='model/1.h5',
                    help='path to save model')
    # ap.add_argument('-pl', '--plot',required=True,
    #                 help='path to save accuracy/loss plot')
    # ap.add_argument('-sm', '--model', required=True,
    #                 help='path to save model')
    args = vars(ap.parse_args())
    # ------------------------------------------------------#
    # 参数
    # ------------------------------------------------------#
    data_path = args['dataset']
    subject = args['subject']
    pretrained = args['pretrained']
    batch_size = int(args['batch_size'])
    epoch = int(args['epoch'])
    # input_shape = (10, 1, 1)
    # input_shape = (16, 8, 1)


    dataset = args['dataset']
    if dataset=="femg":
        classes=38
    elif dataset=='bio':
        classes=6
    else:
        classes = 8
    # subjectList=['001','002']
    totalscore=0

    savePath = 'model/pretrain/' + dataset +'.h5'

    modelFit_TF(savePath)





    # print('pretrain准确率：',score)


    # # 1、加载划分后数据集 subject,data_path,pretrained
    # preData = dataPre(subject, data_path, pretrained)
    # # 2、训练模型 dataPre,input_shape,classes
    # model, score = modelFit(preData, input_shape, classes)

    # trainx=preData[0]
    # trainy = preData[2]
    # prey=model.predict(trainx)
    print('111')


