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
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau,ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, load_model
#选择GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#linux 报错可用：cuDNN launch failure : input shape
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'





#根据epoch动态改变学习率



# predictions = model.predict(x = valX, batch_size=batch_size)
#print(classification_report(valY.argmax(axis=1.h5), predictions.argmax(axis=1.h5), target_names=lb.classes_))
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument('-d', '--dataset', required=True,
    #                 help='path of input dataset of mat')
    # ap.add_argument('-s', '--subject', required=False,
    #                 help='if not pretrained, subject is required, like 001')
    ap.add_argument('-d', '--dataset', default='dbb-1', choices=['dba', 'dbb-1', 'dbb-2', 'femg', 'db7'],
                    help='path of input dataset of mat')
    ap.add_argument('-s', '--subject', default='001',
                    help='if not pretrained, subject is required, like 001')

    ap.add_argument('-p', '--pretrained', action='store_true',
                    help='if pretrained give the parser --pretrained, else not')
    ap.add_argument('-m', '--premodel', default=False,
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
    dataset = args['dataset']
    subject = args['subject']
    pretrained = args['pretrained']
    batch_size = int(args['batch_size'])
    epoch = int(args['epoch'])
    usePreModel=args['premodel']
    # input_shape = (10, 1, 1)
    # input_shape = (16, 8, 1)
    classes=8
    if dataset=='dba':
        data_path='data/capgMyo/DBa'
        subjectList = ['001', '002', '003', '004', '005', '006', '007', '008',
                       '009', '010', '011', '012', '013', '014', '015', '016', '017', '018']
    elif dataset=='dbb-1':
        # data_path = 'data/capgMyo/DBb'
        data_path = 'data/gan/dbb'
        subjectList = ['001', '003',  '005',  '007',
                       '009',  '011',  '013',  '015',  '017', '019']
        testList=['002', '004',  '006',  '008',
                       '010',  '012',  '014',  '016',  '018', '020']
    elif dataset=='dbb-2':
        data_path = 'data/capgMyo/DBb'
        subjectList = ['002', '004', '006', '008',
                    '010', '012', '014', '016', '018', '020']
        testList=['002', '004',  '006',  '008',
                       '010',  '012',  '014',  '016',  '018', '020']
    elif dataset=='femg':
        subjectList = ['001', '002', '003', '004', '005', '006', '007', '008', '009',
                       '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
                       '021', '022', '023', '024', '025', '026', '027', '028']
    # subjectList=['001','002']
    totalscore=0

    # subjectList=['001']
    #检查该文件夹是否存在 没有就创建
    # CheckFolder('model/'+'DBa/')
    scoreList=[]
    for i in tqdm(range(len(subjectList))):
        if usePreModel:
            modelPath='model/'+dataset+'/True/'+subjectList[i]+'.h5'
        else:
            modelPath="model/pretrain/dbb-1.h5"
            # modelPath='model/'+dataset+'/False/'+subjectList[i]+'.h5'
        model=load_model(modelPath)

        # 1、加载划分后数据集 subject,data_path,pretrained
        # preData=dataloader.ICapgMyo_dba_odd_even(data_path,subjectList[i])
        if dataset=="femg":
            preData = dataloader.IFemg(dataset, subjectList[i])
        else:
            preData = dataloader.ICapgMyo_dba_odd_even_slide('dbb-1', testList[i])
            # ganData = dataloader.ICapgMyo_getGanData(dataset, testList[i])
        valX = preData[1]
        valY = preData[3]
        # valX = ganData[0]
        # valY = ganData[1]
        preY=model.predict(valX)
        preY_label = [np.argmax(one_hot) for one_hot in preY]
        valY_label=[np.argmax(one_hot) for one_hot in valY]
        true=0
        for index in range(len(preY_label)):
            if(preY_label[index]==valY_label[index]):
                true=true+1
        score=true/len(valY_label)

        # del preData
        del preY_label
        del valY_label
        # preData=dataPre(subjectList[i],data_path,pretrained)
        #2、训练模型 dataPre,input_shape,classes
        # model,score=modelFit(preData,input_shape,classes,savePath)


        scoreList.append(score)
        print('第%s个被试的准确率为%s' % (i+1, score))
        totalscore+=score
    avgscore=totalscore/len(subjectList)
    print('数据集的平均准确率：',avgscore)
    outPath = 'output/' + dataset + '/2SRNN_test.xls'
    save_excel([subjectList, scoreList], ['subject', 'acc'], outPath)

    # # 1、加载划分后数据集 subject,data_path,pretrained
    # preData = dataPre(subject, data_path, pretrained)
    # # 2、训练模型 dataPre,input_shape,classes
    # model, score = modelFit(preData, input_shape, classes)

    # trainx=preData[0]
    # trainy = preData[2]
    # prey=model.predict(trainx)
    print('111')


