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
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib
import tensorflow as tf
from tqdm import tqdm
from utils import getGpuMrmory, save_excel
import matplotlib
# matplotlib.use('module://backend_interagg')
from imutils import paths
import matplotlib.pyplot as plt
import pickle
import os
import time
from Common import CheckFolder
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier

def Set_GPU():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1' #指定第一块GPU可用
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_list:
        #设置显存不占满
        tf.config.experimental.set_memory_growth(gpu, True)
        #设置显存占用最大值
        tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]
        )

'''
计算CapgMyo DB-a 18个人的训练准确率平均值
'''


# ------------------------------------------------------#
# 数据加载
# ------------------------------------------------------#
def dataPre(subject, data_path, pretrained):
    # 加载data和label
    # data, labels = dataloader.INiaPro_db1(data_path)
    data, labels = dataloader.DataLoader(subject=subject, data_path=data_path, pretrained=pretrained).get_item()
    data = np.array(data, dtype='float')
    labels = np.array(labels)
    # ------------------------------------------------------#
    # 数据划分，test-size:划分比例，训练集80%, 验证集20%
    # ------------------------------------------------------#
    (trainX, valX, trainY, valY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    trainX = np.expand_dims(trainX, axis=3)
    valX = np.expand_dims(valX, axis=3)
    # ------------------------------------------------------#
    # 转为二值类别矩阵
    trainY = tf.keras.utils.to_categorical(trainY)
    # print(trainY)
    valY = tf.keras.utils.to_categorical(valY)
    return trainX, valX, trainY, valY
    # print ("X_train shape: " + str(trainX.shape))
    # print ("Y_train shape: " + str(trainY.shape))
    # print ("X_test shape: " + str(valX.shape))
    # print ("Y_test shape: " + str(valY.shape))


# ------------------------------------------------------#
# 定义model
# ------------------------------------------------------#

def searchBestMode(dataset):
    modelPath = 'model/' + dataset + '/False'
    model_paths = sorted(list(paths.list_files(modelPath)))
    scoreList = []
    for model in model_paths:
        score = model.split('_')[1][:-3]
        scoreList.append(score)
    modelIndex = scoreList.index(max(scoreList))
    bestModelPath = model_paths[modelIndex]
    return bestModelPath


def modelFit_2SRNN(dataPre, classes, savePath,subject):
    # 数据集
    trainX = dataPre[0]
    trainY = dataPre[2]


    valX = dataPre[1]
    valY = dataPre[3]

    if useGan:
        ganData=dataloader.ICapgMyo_getGanData(dataset,subject)
        valX = ganData[0]
        valY = ganData[1]
    input_shape = []
    input_shape.append(trainX.shape[1])
    input_shape.append(trainX.shape[2])
    savePath = 'model/' + dataset + '/' + str(pretrained) + '/' + subjectList[i] + '.h5'
    # 构建模型
    checkpoint = ModelCheckpoint(filepath=savePath, monitor='val_accuracy', mode='auto', save_best_only=True,
                                 save_weights_only=False)
    if pretrained:
        # bestModelPath = searchBestMode(dataset)
        model = NetModel.model_2SRNN_Pretrain(input_shape, classes, dataset)
        # originData = dataloader.ICapgMyo_dba_odd_even_slide(dataset, "001", False)
        # targetData = dataloader.ICapgMyo_dba_odd_even_slide(dataset, "002", False)
        # model = NetModel.model_2SRNN_adapt(input_shape, classes, dataset)


    else:

        model = NetModel.model_2SRNN(input_shape=input_shape, classes=classes)
    # model = NetModel.model_2SRNN_Pretrain(input_shape=input_shape, classes=classes)
    # 打印网络模型结构
    # model.summary()
    # 衰减率
    WEIGHT_DECAY = 0.0001
    # 学习率回调
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # 定义优化器
    # opt = SGD(lr=0.1, decay=0.0001)
    # #模型训练计时
    start = time.time()
    # #模型编译
    # model.compile(loss='CategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])
    # 模型训练
    # H = model.fit(x=trainX, y=trainY, validation_data=(valX, valY),
    #               epochs=epoch, batch_size=batch_size, callbacks=[checkpoint])
    H = model.fit(x=trainX,y=trainY,
                  epochs=epoch, batch_size=batch_size)
    # H = model.fit([originData[0],targetData[0]],originData[2],
    #               epochs=epoch, batch_size=batch_size)
    # 绘制训练loss和acc
    # ------------------------------------------------------#
    # preds_train = model.evaluate(trainX, trainY)
    # print("Train Loss = " + str(preds_train[0]))
    # print("Train Accuracy = " + str(preds_train[1]))
    preds_test = model.evaluate(valX, valY)
    # preds_test = model.evaluate([originData[1],targetData[1]], targetData[3])
    print("Test Loss = " + str(preds_test[0]))
    print("Test Accuracy = " + str(preds_test[1]))
    end = time.time()
    print("模型训练时长:", end - start, "s")
    # 模型保存
    # model.save(savePath, save_format="h5")

    # 绘制loss和acc图
    # N = np.arange(0, epoch)
    # plt.style.use('ggplot')
    # plt.figure()
    # plt.plot(N, H.history['loss'], 'g', label='train_loss')
    # plt.plot(N, H.history['val_loss'], 'k', label='val_loss')
    # plt.plot(N, H.history['accuracy'], 'r', label='train_acc')
    # plt.plot(N, H.history['val_accuracy'], 'b', label='val_acc')
    # plt.title("Training Loss and Accuracy (Simple NN)")
    # plt.xlabel("Epoch:#" + str(epoch))
    # plt.ylabel("Loss/Accuracy")
    # plt.legend()
    # plt.savefig(args["plot"])
    del model
    return preds_test[1]


def modelFit(dataPre, input_shape, classes, savePath):
    # 数据集
    trainX = dataPre[0]
    valX = dataPre[1]
    trainY = dataPre[2]
    valY = dataPre[3]
    # 构建模型
    model = NetModel.buile_model(input_shape=input_shape, classes=classes)
    # 打印网络模型结构
    # model.summary()
    # 衰减率
    WEIGHT_DECAY = 0.0001
    # 学习率回调
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # 定义优化器
    opt = SGD(lr=0.1, decay=0.0001)
    # 模型训练计时
    start = time.time()
    # 模型编译
    model.compile(loss='CategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])
    # 模型训练
    H = model.fit(x=trainX, y=trainY, validation_data=(valX, valY), epochs=epoch, batch_size=batch_size,
                  callbacks=[reduce_lr])

    # 绘制训练loss和acc
    # ------------------------------------------------------#
    preds_train = model.evaluate(trainX, trainY)
    print("Train Loss = " + str(preds_train[0]))
    print("Train Accuracy = " + str(preds_train[1]))
    preds_test = model.evaluate(valX, valY)
    print("Test Loss = " + str(preds_test[0]))
    print("Test Accuracy = " + str(preds_test[1]))
    end = time.time()
    print("模型训练时长:", end - start, "s")
    # 模型保存
    model.save(savePath, save_format="h5")

    # 绘制loss和acc图
    # N = np.arange(0, epoch)
    # plt.style.use('ggplot')
    # plt.figure()
    # plt.plot(N, H.history['loss'], 'g', label='train_loss')
    # plt.plot(N, H.history['val_loss'], 'k', label='val_loss')
    # plt.plot(N, H.history['accuracy'], 'r', label='train_acc')
    # plt.plot(N, H.history['val_accuracy'], 'b', label='val_acc')
    # plt.title("Training Loss and Accuracy (Simple NN)")
    # plt.xlabel("Epoch:#" + str(epoch))
    # plt.ylabel("Loss/Accuracy")
    # plt.legend()
    # plt.savefig(args["plot"])

    return model, preds_test[1]


# 根据epoch动态改变学习率
def lr_schedule(epoch):
    lr = 1e-1
    if (epoch >= 16) & (epoch < 24):
        lr = 1e-2
    elif epoch >= 24:
        lr = 1e-3
    print('Learning rate: ', lr)
    return lr


# predictions = model.predict(x = valX, batch_size=batch_size)
# print(classification_report(valY.argmax(axis=1.h5), predictions.argmax(axis=1.h5), target_names=lb.classes_))
if __name__ == '__main__':

    Set_GPU()
    ap = argparse.ArgumentParser()
    # ap.add_argument('-d', '--dataset', required=True,
    #                 help='path of input dataset of mat')
    # ap.add_argument('-s', '--subject', required=False,
    #                 help='if not pretrained, subject is required, like 001')
    ap.add_argument('-d', '--dataset', default='dbb-2', choices=['dba', 'dbb-1', 'dbb-2', 'dbb-all', 'femg','bio'],
                    help='path of input dataset of mat')
    ap.add_argument('-s', '--subject', default='001',
                    help='if not pretrained, subject is required, like 001')

    ap.add_argument('-p', '--pretrained', default=True,
                    help='if pretrained give the parser --pretrained, else not')
    ap.add_argument('-g', '--gan', default=False,
                    help='if use gan emg test --gan, else not')
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
    useGan=args['gan']
    # input_shape = (10, 1, 1)
    # input_shape = (16, 8, 1)

    if dataset == 'dba':
        data_path = 'data/capgMyo/DBa'
        subjectList = ['001', '002', '003', '004', '005', '006', '007', '008',
                       '009', '010', '011', '012', '013', '014', '015', '016', '017', '018']
        classes = 8
    elif dataset == 'dbb-1':
        data_path = 'data/capgMyo/DBb'
        subjectList = ['001', '003', '005', '007',
                       '009', '011', '013', '015', '017', '019']
        classes = 8
    elif dataset == 'dbb-2':
        data_path = 'data/capgMyo/DBb'
        # subjectList = ['002', '004', '006', '008',
        #                '010', '012', '014', '016', '018', '020']
        subjectList = ['002']
        classes = 8
    elif dataset == 'dbb-all':
        data_path = 'data/capgMyo/DBb'
        subjectList = ['001', '002', '003', '004', '005', '006', '007', '008', '009',
                       '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020']
        classes = 8
    elif dataset=='femg':
        data_path = 'data/femg'
        subjectList = ['001','002', '003', '004', '005', '006', '007', '008', '009',
                       '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
                       '021','022','023','024','025','026','027','028']
        # subjectList = [ '018', '019', '020', '021', '022', '023', '024',
        #                '025', '026', '027', '028']
        classes = 38
    elif dataset=='bio':
        data_path = 'data/bio'
        subjectList = ['000','001', '002', '003', '004', '005', '006', '007', '008', '009']
        # subjectList = ['000']
        # subjectList = [ '018', '019', '020', '021', '022', '023', '024',
        #                '025', '026', '027', '028']
        classes = 6
    # subjectList=['001','002']
    totalscore = 0

    # subjectList=['001']
    # 检查该文件夹是否存在 没有就创建
    # CheckFolder('model/'+'DBa/')
    scoreList = []
    for i in tqdm(range(len(subjectList))):
        savePath = 'model/' + dataset + '/' + subjectList[i] + '.h5'
        # 1、加载划分后数据集 subject,data_path,pretrained
        # preData=dataloader.ICapgMyo_dba_odd_even(data_path,subjectList[i])
        if dataset=="femg":
            preData = dataloader.IFemg(dataset, subjectList[i])
        elif dataset=='bio':
            preData = dataloader.IBio(subjectList[i])
        else:
            preData = dataloader.ICapgMyo_dba_odd_even_slide(dataset, subjectList[i], True)
        # preData=dataPre(subjectList[i],data_path,pretrained)
        # 2、训练模型 dataPre,input_shape,classes
        # model,score=modelFit(preData,input_shape,classes,savePath)
        score = modelFit_2SRNN(preData, classes, savePath,subjectList[i])

        scoreList.append(score)
        print('第%s个被试的准确率为%s' % (i + 1, score))
        totalscore += score
    avgscore = totalscore / len(subjectList)
    print('数据集的平均准确率：', avgscore)
    outPath = 'output/' + dataset + '/2SRNN.xls'
    save_excel([subjectList, scoreList], ['subject', 'acc'], outPath)

    # # 1、加载划分后数据集 subject,data_path,pretrained
    # preData = dataPre(subject, data_path, pretrained)
    # # 2、训练模型 dataPre,input_shape,classes
    # model, score = modelFit(preData, input_shape, classes)

    # trainx=preData[0]
    # trainy = preData[2]
    # prey=model.predict(trainx)
    print('111')
