import numpy as np
import os
from imutils import paths
from tqdm import tqdm
import random
import mat4py
import platform
import scipy.io as scio
from scipy.signal import butter, lfilter, filtfilt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from utils import standSc
import gc
# def normalization(semg_data):
#     #将semg信号从[-2.5mv, 2.5mv]归一化到[0, 1.h5]之间
#     return (semg_data-(-2.5))/(2.5-(-2.5))
#读取CapgMyo数据集
class DataLoader():
    def __init__(self, subject, data_path, pretrained = True):
      self.data = []
      self.labels = []
      self.mats = []

      mat_paths = sorted(list(paths.list_files(data_path)))
      random.seed(42)
      random.shuffle(mat_paths)

      if pretrained:
        for mat_path in tqdm(mat_paths):
          try:
            # if (platform.system() == 'Linux'):
            #   mat_data = mat4py.loadmat(os.path.join(data_path, mat_path))
            # else:
              mat_data = mat4py.loadmat(mat_path)
          except:
            continue
          data = mat_data['data']
          for line in data:
            new_line = []
            for semg_data in line:
              # img_data = normalization(semg_data)
              # new_line.append(255.0 * img_data)
              new_line.append(semg_data)
            self.data.append(np.reshape(np.array(new_line), (16, 8)))
            self.labels.append(int(mat_path.split('-')[1]) - 1)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
      else:
        for mat_path in tqdm(mat_paths):
          if(platform.system()=='Linux'):
            mat_subject=mat_path.split('-')[0].split('/')[-1]
          else:
            mat_subject = mat_path.split('\\')[1].split('-')[0]
            # mat_path=mat_path.replace("\\","/")
          if mat_subject == subject:
            try:
                mat_data = mat4py.loadmat(mat_path)

            except:
              continue
            data = mat_data['data']

            for line in data:
              new_line = []
              for semg_data in line:
                new_line.append(255.0*semg_data)
              self.data.append(np.reshape(np.array(new_line), (16, 8)))
              self.labels.append(int(mat_path.split('-')[1]) - 1)
              
    def get_item(self):
      return self.data, self.labels


def get_segments_image(data, window, stride):
  chnum = data.shape[1];
  data = windowed_view(
    data.flat,
    window * data.shape[1],
    (window - stride) * data.shape[1]
  )
  data = data.reshape(-1, window, chnum)
  # data=data*255
  return data


def windowed_view(arr, window, overlap):
  from numpy.lib.stride_tricks import as_strided
  arr = np.asarray(arr)
  window_step = window - overlap
  new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                window)
  new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                 arr.strides[-1:])
  return as_strided(arr, shape=new_shape, strides=new_strides)
#获得奇数次数和偶数的数据和标签


def IGAN_emg_All(ninapro,subjectList):
  # subjectList = ['000', '001', '002', '003','004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
  #                '014','015','016','017','018','019','021']
  # subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
  # subjectList = ['000']
  # subjectList = ['000', '001', '002', '003', '005', '006', '007', '008', '009', '010']
  # subjectList =['000']
  # subjectList = ['000']
  # subjectList = ['000', '001', '002', '003', '004', '005','006','007','008','009','010'
  #                ,'011','012','013','014','015','016','017','018','019'
  #                ]
  emgDataList=[]
  trainLabelList=[]
  for i in range(len(subjectList)):
    emgData,label=IGAN_emg_single(ninapro,subjectList[i],-1,-1)
    emgDataList.append(emgData)
    trainLabelList.append(label)
    del emgData
    del label
    gc.collect()
  emgDataArry=np.vstack(emgDataList)
  trainLabelArry=np.hstack(trainLabelList)
  return emgDataArry,trainLabelArry


def IGAN_emg_single(dataset, subject, gesture, repnum):

  if (dataset == 'dbb-1')|(dataset == 'dbb-2'):
    # 训练集 第1、3,4,6,8,9,10
    trainindex = [ '001', '003', '005','007','009']
    # testindex = ['001', '004']
    classes = 8
    channel = 16
    ge_first_index = 1
    lowpass_flag = False
    downsample_flag=False
  elif dataset == 'db1':
    trainindex = ['000', '002', '003', '005', '007', '008', '009']
    testindex = ['001', '004', '006']
    classes = 52
    channel = 10
    ge_first_index = 1
    lowpass_flag = False
  emgData = []
  imuData = []
  trainLabel = []
  # testLabel = []
  emgPaths = []
  imuPaths = []
  # testPaths = []
  # data/semg_data/ninapro_db1/001
  if (dataset=="dbb-1")|(dataset=="dbb-2"):
    emgPath='data/capgMyo/DBb/'
  else:
    emgPath = 'data/ninapro_' + str(dataset) + '/' + str(subject)

  # 生成训练集和测试集路径
  # select specify gesture
  if gesture != -1:
    geindex = str(int(gesture) + ge_first_index).rjust(3, '0')
    emgPath_1 = emgPath
    # select specify rep
    if repnum != -1:
      filePath1 = emgPath_1 + subject + '-' + geindex + '-' + repnum + '.mat'
      emgPaths.append(filePath1)
    else:
      for j in range(len(trainindex)):
        filePath1 = emgPath_1 + subject + '-' + geindex + '-' + trainindex[j] + '.mat'
        emgPaths.append(filePath1)
  else:
    for i in tqdm(range(classes)):
      geindex = str(i + ge_first_index).rjust(3, '0')
      emgPath_1 = emgPath
      # 生成训练集文件路径
      for j in range(len(trainindex)):
        filePath1 = emgPath_1 + subject + '-' + geindex + '-' + trainindex[j] + '.mat'
        emgPaths.append(filePath1)
  # 获取训练集数据和标签
  for emgPath in emgPaths:
    f = scio.loadmat(emgPath)
    # label = f['label'][0][0] - 1
    label = f['gesture'][0][0] - ge_first_index
    data = f['data'].astype(np.float32)
    if downsample_flag:
      data = downsample(data, step=20)
    for x in range(data.shape[0]):
      sigData = data[x]
      emgData.append(sigData)
      trainLabel.append(label)
  # 获取测试集数据和标签

  emgDataArry = np.array(emgData, dtype=np.float32)
  trainLabelArry = np.array(trainLabel, dtype=np.int32)
  return emgDataArry, trainLabelArry


def ICapgMyo_getGanData(dataset,subject):

  dataList = []
  labelList = []

  #获得datapath下所有文件的路径
  slide = 70
  meanOf = 11

  if dataset=="dba":
    dataPath='data/capgMyo/DBa'
  elif dataset=='dbb-1':
    # dataPath = 'data/capgMyo/DBb'
    dataPath = 'data/gan/dbb'
  elif dataset=='dbb-2':
    dataPath = 'data/gan/dbb'
  elif dataset=='dbb-all':
    dataPath = 'data/capgMyo/DBb'
  mat_paths = sorted(list(paths.list_files(dataPath)))


  for mat_path in tqdm(mat_paths):
    mat_subject = mat_path.split('-')[0][-3:]
    classes=mat_path.split('-')[1][-3:]
    repeat = mat_path.split('-')[2][0:3]
    #筛选指定测试人
    if (mat_subject==subject)&((repeat=='002')|(repeat=='004')|(repeat=='006')|(repeat=='008')|(repeat=='010')):
      f = scio.loadmat(mat_path)
      data = f['data'].astype(np.float32)
      # data = (data - mean) / std
      data=standSc(data)
      data = np.abs(data)
      data = np.apply_along_axis(lambda m: np.convolve(m, np.ones((meanOf,))/meanOf, mode='valid'), axis=0, arr=data)
      # 同乘255 转为灰度图像
      # data = data * 255.0
      label=f['gesture'][0][0]-1
      #获得测试次数索引
      data=get_segments_image(data,150,slide)
      dataList.append(data)
      labelList.append([label]*data.shape[0])

  DataArry = np.array(dataList, dtype='float')

  LabelArry = np.array(labelList)
  del dataList
  del labelList



  DataArry=DataArry.reshape([DataArry.shape[0]*DataArry.shape[1]]+list(DataArry.shape[2:]))

  LabelArry = LabelArry.reshape(
    [LabelArry.shape[0] * LabelArry.shape[1]])

  # ------------------------------------------------------#
  # 转为二值类别矩阵
  LabelArry = tf.keras.utils.to_categorical(LabelArry)
  return DataArry, LabelArry

def ICapgMyo_dba_odd_even_slide(dataset,subject,oneRepeat=False):

  dataOddList = []
  labelOddlList = []
  dataEvenList = []
  labelEvenlList = []
  #获得datapath下所有文件的路径
  slide = 70
  meanOf = 11

  if dataset=="dba":
    mean = -0.00045764610078446837
    std = 0.04763047257531886
    dataPath='data/capgMyo/DBa'
  elif dataset=='dbb-1':
    mean = -1.494554769724577e-06
    std = 0.014456551081024154
    dataPath = 'data/capgMyo/DBb'
  elif dataset=='dbb-2':
    mean = -3.342415416859754e-07
    std = 0.012518382863642336
    dataPath = 'data/capgMyo/DBb'
  elif dataset=='dbb-all':
    mean = -3.342415416859754e-07
    std = 0.012518382863642336
    dataPath = 'data/capgMyo/DBb'
  mat_paths = sorted(list(paths.list_files(dataPath)))


  for mat_path in tqdm(mat_paths):
    mat_subject = mat_path.split('-')[0][-3:]
    repeat=mat_path.split('-')[1][-3:]
    #筛选指定测试人
    if mat_subject==subject:
      f = scio.loadmat(mat_path)
      data = f['data'].astype(np.float32)
      # data = (data - mean) / std
      data=standSc(data)
      data = np.abs(data)
      data = np.apply_along_axis(lambda m: np.convolve(m, np.ones((meanOf,))/meanOf, mode='valid'), axis=0, arr=data)
      # 同乘255 转为灰度图像
      # data = data * 255.0
      label=f['gesture'][0][0]-1
      #获得测试次数索引
      index = int(mat_path.split('-')[2][0:3])
      data=get_segments_image(data,150,slide)

          # 测试次数为偶数
      if index%2==0:

        dataEvenList.append(data)
        labelEvenlList.append([label]*data.shape[0])
      else:
        if oneRepeat:
          if (index==1):
            dataOddList.append(data)
            labelOddlList.append([label] * data.shape[0])
        else:
          dataOddList.append(data)
          labelOddlList.append([label]*data.shape[0])

  trainDataArry = np.array(dataOddList, dtype='float')
  testDataArry = np.array(dataEvenList, dtype='float')
  trainLabelArry = np.array(labelOddlList)
  testLabelArry = np.array(labelEvenlList)
  del dataOddList
  del dataEvenList
  del labelOddlList
  del labelEvenlList


  trainDataArry=trainDataArry.reshape([trainDataArry.shape[0]*trainDataArry.shape[1]]+list(trainDataArry.shape[2:]))
  testDataArry = testDataArry.reshape(
    [testDataArry.shape[0] * testDataArry.shape[1]] + list(testDataArry.shape[2:]))
  trainLabelArry = trainLabelArry.reshape(
    [trainLabelArry.shape[0] * trainLabelArry.shape[1]])
  testLabelArry = testLabelArry.reshape(
    [testLabelArry.shape[0] * testLabelArry.shape[1]])
  # trainDataArry = np.expand_dims(trainDataArry, axis=3)
  # testDataArry = np.expand_dims(testDataArry, axis=3)
  # ------------------------------------------------------#
  # 转为二值类别矩阵
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry, testDataArry, trainLabelArry, testLabelArry

def downsample(data, step):
  return data[::step].copy()


def emg_ssi(signal):
  signal_squ = [s * s for s in signal]
  signal_squ = np.array(signal_squ)
  return np.sum(signal_squ)

def emg_rms(signal):
  signal = np.array(signal)
  ssi = emg_ssi(signal)
  length = signal.shape[0]
  if length <= 0:
    return 0
  return np.sqrt(float(ssi) / length)
def preProcess(filePath,ds):
  emg = scio.loadmat(filePath)
  if ds=='femg':
    abs_flag=True
    butter_flag=False
    downsample_flag=True
    normalize_flag=True
    rms_flag=False
    window = 150
    stride = 70
    framerate = 2048
    ge_first_index=0
    label = emg['gesture'][0][0] - ge_first_index
    emg = emg['HDemg'].astype(np.float32)
  elif ds=='bio':
    abs_flag = True
    butter_flag = False
    downsample_flag = True
    normalize_flag = False
    rms_flag=False
    window = 150
    stride = 70
    # window = 600
    # stride = 300
    framerate = 4000
    ge_first_index = 0
    label = emg['label'][0][0] - ge_first_index
    emg = emg['data'].astype(np.float32)


  if abs_flag:
    emg = np.abs(emg)
    # imu = np.abs(imu)
  if butter_flag:
    # pass
    # emg = np.transpose([butter_filter(ch, 20, 450, 2000, 10, zero_phase=True) for ch in emg.T])
    emg = np.transpose([butter_lowpass_filter(ch, 1, framerate, 1, zero_phase=True) for ch in emg.T])
    # imu = np.transpose([butter_lowpass_filter(ch, 1, framerate, 1, zero_phase=True) for ch in imu.T])
  if downsample_flag:
    emg = downsample(emg, step=10)

  if normalize_flag:
    emg = standSc(emg)
    # imu = min_max(imu, -1.0, 1.0)
  if rms_flag:
    emg=emg_rms(emg)
  emg=get_segments_image(emg, window, stride)

  labelList = [label] * emg.shape[0]
  return emg,labelList

def IBio_session(subject):
#data/semg_data/ninapro_db1/001
  sessionIndex=['001','002']
  trainindex = [0,1]
  testindex = [2]
  emgPath = 'data/bio/' + str(subject)
  trainData = []
  testData = []
  trainLabel = []
  testLabel = []
  ge_first_index=0
  classes=6
  for i in range(0, 3):
  # class
    for j in range(0, classes):
        for train in sessionIndex:
          for k in trainindex:
            # filePath = emgPath + '/' + subject + '_' + geindex + '_' + trainindex[j] + '.mat'
            filePath_emg = emgPath+'/{0:03d}/{1:03d}/{2}/{4}_{0:03d}_{1:03d}_{2}_{3:03d}.mat'.format(i+1,j+ge_first_index,train,k,subject)

            res, labelList = preProcess(filePath_emg, "bio")
            trainData.append(res)
            trainLabel.extend(labelList)
        for test in sessionIndex:
          for k in testindex:
            # filePath = emgPath + '/' + subject + '_' + geindex + '_' + trainindex[j] + '.mat'
            filePath_emg = emgPath + '/{0:03d}/{1:03d}/{2}/{4}_{0:03d}_{1:03d}_{2}_{3:03d}.mat'.format(i + 1,
                                                                                                           j + ge_first_index,
                                                                                                           test,k,
                                                                                                           subject)
            res, labelList = preProcess(filePath_emg, "bio")
            testData.append(res)
            testLabel.extend(labelList)
  trainDataArry = np.concatenate(tuple(trainData), axis=0)
  testDataArry = np.concatenate(tuple(testData), axis=0)
  trainLabelArry = np.array(trainLabel, dtype=np.int32)
  testLabelArry = np.array(testLabel, dtype=np.int32)
  # trainDataArry = np.expand_dims(trainDataArry, axis=3)
  # testDataArry = np.expand_dims(testDataArry, axis=3)
  # trainDataArry = trainDataArry.transpose(0, 2, 3, 1)
  # testDataArry = testDataArry.transpose(0, 2, 3, 1)
  # ------------------------------------------------------#
  # 转为二值类别矩阵
  del trainData
  del testData
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry, testDataArry, trainLabelArry, testLabelArry

def IBio(subject):
#data/semg_data/ninapro_db1/001
  trainindex = ['003']
  testindex = ['003']
  emgPath = 'data/bio/' + str(subject)
  trainData = []
  testData = []
  trainLabel = []
  testLabel = []
  ge_first_index=0
  classes=6
  for i in range(0, 3):
  # class
    for j in range(0, classes):
        for train in trainindex:
          for k in range(0, 1):
            # filePath = emgPath + '/' + subject + '_' + geindex + '_' + trainindex[j] + '.mat'
            filePath_emg = emgPath+'/{0:03d}/{1:03d}/{2}/{4}_{0:03d}_{1:03d}_{2}_{3:03d}.mat'.format(i+1,j+ge_first_index,train,k,subject)

            res, labelList = preProcess(filePath_emg, "bio")
            trainData.append(res)
            trainLabel.extend(labelList)
        for test in testindex:
          for k in range(0, 3):
            # filePath = emgPath + '/' + subject + '_' + geindex + '_' + trainindex[j] + '.mat'
            filePath_emg = emgPath + '/{0:03d}/{1:03d}/{2}/{4}_{0:03d}_{1:03d}_{2}_{3:03d}.mat'.format(i + 1,
                                                                                                           j + ge_first_index,
                                                                                                           test,k,
                                                                                                           subject)
            res, labelList = preProcess(filePath_emg, "bio")
            testData.append(res)
            testLabel.extend(labelList)
  trainDataArry = np.concatenate(tuple(trainData), axis=0)
  testDataArry = np.concatenate(tuple(testData), axis=0)
  trainLabelArry = np.array(trainLabel, dtype=np.int32)
  testLabelArry = np.array(testLabel, dtype=np.int32)
  # trainDataArry = np.expand_dims(trainDataArry, axis=3)
  # testDataArry = np.expand_dims(testDataArry, axis=3)
  # trainDataArry = trainDataArry.transpose(0, 2, 3, 1)
  # testDataArry = testDataArry.transpose(0, 2, 3, 1)
  # ------------------------------------------------------#
  # 转为二值类别矩阵
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry, testDataArry, trainLabelArry, testLabelArry
def IFemg(ds,subject):
  if ds=='femg':
    # 训练集 第1、3,4,6,8,9,10
    trainindex = ['005']
    testindex = ['006']
    classes=38

    ge_first_index=0
    emgPath = 'data/femg/'+ str(subject)
  trainData = []
  testData = []
  trainLabel = []
  testLabel = []

  #data/semg_data/ninapro_db1/001



  #生成训练集和测试集路径
  for i in tqdm(range(classes)):
    # geindex=str(i+1).rjust(3, '0')
    geindex=str(i+ge_first_index).rjust(3, '0')
    emgPath_ge=emgPath+'/'+geindex
    #生成训练集文件路径
    for j in range(len(trainindex)):
      filePath=emgPath_ge+'/'+subject+'_'+geindex+'_'+trainindex[j]+'.mat'

      res,labelList=preProcess(filePath,ds)
      trainData.append(res)
      trainLabel.extend(labelList)
      #生成测试集文件路径
    for k in range(len(testindex)):
      filePath = emgPath_ge + '/' + subject + '_' + geindex + '_' + testindex[k] + '.mat'

      res, labelList = preProcess(filePath, ds)
      testData.append(res)
      testLabel.extend(labelList)
  trainDataArry = np.concatenate(tuple(trainData), axis=0)
  # trainDataArry=trainDataArry[:,:,0:1]
  testDataArry = np.concatenate(tuple(testData), axis=0)
  # testDataArry = testDataArry[:, :, 0:1]
  trainLabelArry = np.array(trainLabel,dtype=np.int32)
  testLabelArry = np.array(testLabel,dtype=np.int32)

  # trainDataArry = np.expand_dims(trainDataArry, axis=3)
  # testDataArry = np.expand_dims(testDataArry, axis=3)
  # trainDataArry =trainDataArry.transpose(0,2,3,1)
  # testDataArry = testDataArry.transpose(0, 2, 3, 1)

  # trainDataArry = trainDataArry.transpose(0, 2, 3, 1)
  # testDataArry = testDataArry.transpose(0, 2, 3, 1)

  # ------------------------------------------------------#
  # 转为二值类别矩阵
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry,testDataArry,trainLabelArry,testLabelArry


def ICapgMyo_dba_odd_even_slide_session(dataset,subject):

  dataList = []
  labelList = []

  #获得datapath下所有文件的路径
  slide = 70
  meanOf = 11

  if dataset=="dba":
    mean = -0.00045764610078446837
    std = 0.04763047257531886
    dataPath='data/capgMyo/DBa'
  elif dataset=='dbb-1':
    mean = -1.494554769724577e-06
    std = 0.014456551081024154
    dataPath = 'data/capgMyo/DBb'
  elif dataset=='dbb-2':
    mean = -3.342415416859754e-07
    std = 0.012518382863642336
    dataPath = 'data/capgMyo/DBb'
  elif dataset=='dbb-all':
    mean = -3.342415416859754e-07
    std = 0.012518382863642336
    dataPath = 'data/capgMyo/DBb'
  elif dataset=='dbb-all-session':
    mean = -3.342415416859754e-07
    std = 0.012518382863642336
    dataPath = 'data/capgMyo/DBb'
  mat_paths = sorted(list(paths.list_files(dataPath)))


  for mat_path in tqdm(mat_paths):
    mat_subject = mat_path.split('-')[0][-3:]
    #筛选指定测试人
    if mat_subject==subject:
      f = scio.loadmat(mat_path)
      data = f['data'].astype(np.float32)
      data = (data - mean) / std
      # data=standSc(data)
      data = np.abs(data)
      data = np.apply_along_axis(lambda m: np.convolve(m, np.ones((meanOf,))/meanOf, mode='valid'), axis=0, arr=data)
      # 同乘255 转为灰度图像
      # data = data * 255.0
      label=f['gesture'][0][0]-1
      #获得测试次数索引
      index = int(mat_path.split('-')[2][0:3])
      data=get_segments_image(data,150,slide)
      dataList.append(data)
      labelList.append([label]*data.shape[0])


  DataArry = np.array(dataList, dtype='float')

  LabelArry = np.array(labelList)

  del dataList
  del labelList



  DataArry=DataArry.reshape([DataArry.shape[0]*DataArry.shape[1]]+list(DataArry.shape[2:]))

  LabelArry = LabelArry.reshape(
    [LabelArry.shape[0] * LabelArry.shape[1]])

  # trainDataArry = np.expand_dims(trainDataArry, axis=3)
  # testDataArry = np.expand_dims(testDataArry, axis=3)
  # ------------------------------------------------------#
  # 转为二值类别矩阵

  LabelArry = tf.keras.utils.to_categorical(LabelArry)
  return DataArry, LabelArry

#获得奇数次数和偶数的数据和标签
#获得奇数次数和偶数的数据和标签
def ICapgMyo_dba_odd_even(dataPath,subject):

  dataOddList = []
  labelOddlList = []
  dataEvenList = []
  labelEvenlList = []
  #获得datapath下所有文件的路径
  mat_paths = sorted(list(paths.list_files(dataPath)))


  for mat_path in tqdm(mat_paths):
    mat_subject = mat_path.split('-')[0][-3:]
    #筛选指定测试人
    if mat_subject==subject:
      f = scio.loadmat(mat_path)
      data = f['data'].astype(np.float32)
      # 同乘255 转为灰度图像
      data = data * 255.0
      label=f['gesture'][0][0]-1
      #获得测试次数索引
      index = int(mat_path.split('-')[2][0:3])
      for line in data:

          # 测试次数为偶数
        if index%2==0:
          dataEvenList.append(np.reshape(line, (16, 8)))
          labelEvenlList.append(label)
        else:
          dataOddList.append(np.reshape(line, (16, 8)))
          labelOddlList.append(label)

  trainDataArry = np.array(dataOddList, dtype='float')
  testDataArry = np.array(dataEvenList, dtype='float')
  trainLabelArry = np.array(labelOddlList)
  testLabelArry = np.array(labelEvenlList)

  trainDataArry = np.expand_dims(trainDataArry, axis=3)
  testDataArry = np.expand_dims(testDataArry, axis=3)
  # ------------------------------------------------------#
  # 转为二值类别矩阵
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry, testDataArry, trainLabelArry, testLabelArry


#获得奇数次数和偶数的数据和标签

def min_max(data, min, max):
      """Normalize data"""
      scaler = MinMaxScaler(feature_range=(min, max),copy=True)
      scaler.fit(data)
      norm_value = scaler.transform(data)
      return [norm_value, scaler]

#获得奇数次数和偶数的数据和标签
def ICapgMyo_dba_odd_even2(dataPath,subject):

  dataOddList = []
  labelOddlList = []
  dataEvenList = []
  labelEvenlList = []
  #获得datapath下所有文件的路径
  mat_paths = sorted(list(paths.list_files(dataPath)))


  for mat_path in tqdm(mat_paths):
    mat_subject = mat_path.split('-')[0][-3:]
    #筛选指定测试人
    if mat_subject==subject:
      f = scio.loadmat(mat_path)
      data = f['data'].astype(np.float32)

      data=min_max(data,0,1)[0]
      # 同乘255 转为灰度图像
      data = data * 255.0
      label=f['gesture'][0][0]-1
      #获得测试次数索引
      index = int(mat_path.split('-')[2][0:3])
      for line in data:

          # 测试次数为偶数
        if index%2==0:
          dataEvenList.append(np.reshape(line, (16, 8)))
          labelEvenlList.append(label)
        else:
          dataOddList.append(np.reshape(line, (16, 8)))
          labelOddlList.append(label)

  trainDataArry = np.array(dataOddList, dtype='float')
  testDataArry = np.array(dataEvenList, dtype='float')
  trainLabelArry = np.array(labelOddlList)
  testLabelArry = np.array(labelEvenlList)

  trainDataArry = np.expand_dims(trainDataArry, axis=3)
  testDataArry = np.expand_dims(testDataArry, axis=3)
  # ------------------------------------------------------#
  # 转为二值类别矩阵
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry, testDataArry, trainLabelArry, testLabelArry


'''对于INiaPro 数据集采用butter-worth低通滤波预处理
参数：数据，
'''
def butter_lowpass_filter(data, cut, fs, order, zero_phase=False):
  nyq = 0.5 * fs
  cut = cut / nyq

  b, a = butter(order, cut, btype='low')
  y = (filtfilt if zero_phase else lfilter)(b, a, data)
  return y

#INiaPro 做低通滤波
def INiaPro_butter(data):
  dataArry = []
  # framerate为信号采样率，NinaProDB1为100Hz，其
  for i in range(data.shape[1]):
    chData = data[:, i]
    butter_chData = butter_lowpass_filter(chData, 1, 100,1,True)
    dataArry.append(butter_chData)
  dataArry = np.array(dataArry)
  # 行列互换
  butterdata = dataArry.T
  return butterdata

#加载NiaPro数据（加载一个被试的整个数据）
def INiaPro_db1(dataPath,butterFlag):
  dataList = []
  labelList = []
  mat_paths = sorted(list(paths.list_files(dataPath)))
  # mat_data = mat4py.loadmat('data/semg_data/ninapro_db1/000/000/000_000_000.mat')
  # f = scio.loadmat('data/semg_data/ninapro_db1/000/000/000_000_000.mat')
  # f=mat4py.loadmat('data/semg_data/ninapro_db1/000/000/000_000_000.mat')
  for mat_path in tqdm(mat_paths):
    f = scio.loadmat(mat_path)
    label = f['label'][0][0] - 1
    data = f['data'].astype(np.float32)
    if butterFlag:
      data = np.transpose([butter_lowpass_filter(ch, 1, 100, 1, zero_phase=True) for ch in data.T])
      # data=INiaPro_butter(data)

    for i in range(data.shape[0]):
      dataList.append(np.reshape(data[i], (10, 1)))
      labelList.append(label)

  return dataList,labelList
  # data = f['features']

'''加载NiaPro_db1数据并按指定索引划分训练测试集
参数：数据集路径，被试编号，低通滤波标志
'''
def INiaPro_db1_split(datapath,subject,butterFlag):
  #data/semg_data/ninapro_db1/001
  datapath=datapath+'/'+subject
  #训练集 第1、3,4,6,8,9,10
  trainindex=['000','002','003','005','007','008','009']
  testindex=['001','004','006']
  trainData=[]
  testData=[]
  trainLabel=[]
  testLabel=[]
  trainPaths = []
  testPaths = []
  #生成训练集和测试集路径
  for i in range(52):
    geindex=str(i+1).rjust(3, '0')
    path1=datapath+'/'+geindex
    #生成训练集文件路径
    for j in range(len(trainindex)):
      filePath=path1+'/'+subject+'_'+geindex+'_'+trainindex[j]+'.mat'
      trainPaths.append(filePath)
    #生成测试集文件路径
    for k in range(len(testindex)):
      filePath=path1+'/'+subject+'_'+geindex+'_'+testindex[k]+'.mat'
      testPaths.append(filePath)


  #获取训练集数据和标签
  for train_path in tqdm(trainPaths):
    f = scio.loadmat(train_path)
    label = f['label'][0][0] - 1
    data = f['data'].astype(np.float32)
    if butterFlag:
      data = np.transpose([butter_lowpass_filter(ch, 1, 100, 1, zero_phase=True) for ch in data.T])
      # data=INiaPro_butter(data)
    for i in range(data.shape[0]):
      trainData.append(np.reshape(data[i], (10, 1)))
      trainLabel.append(label)
  # 获取测试集数据和标签
  for test_path in tqdm(testPaths):
    f = scio.loadmat(test_path)
    label = f['label'][0][0] - 1
    data = f['data'].astype(np.float32)
    if butterFlag:
      data = np.transpose([butter_lowpass_filter(ch, 1, 100, 1, zero_phase=True) for ch in data.T])

    for i in range(data.shape[0]):
      testData.append(np.reshape(data[i], (10, 1)))
      testLabel.append(label)

  trainDataArry = np.array(trainData, dtype='float')
  testDataArry = np.array(testData, dtype='float')
  trainLabelArry = np.array(trainLabel)
  testLabelArry = np.array(testLabel)

  trainDataArry = np.expand_dims(trainDataArry, axis=1)
  testDataArry = np.expand_dims(testDataArry, axis=1)
  # ------------------------------------------------------#
  # 转为二值类别矩阵
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry,testDataArry,trainLabelArry,testLabelArry

  print('1111')
  # mat_paths = sorted(list(paths.list_files(dataPath)))

def getDataDictCount(filePath):
  myDict=np.load(filePath,allow_pickle='TRUE').item()
  countSum=sum(myDict.values())
  return countSum

def GetTfFilesList(mode,ninapro):
  fileList=[]
  if ninapro=='db1':
    subNum=27
  elif ninapro=='db5':
    subNum=10
  elif ninapro=='db2':
    subNum=40
  elif ninapro=='db3':
    subNum=6
  elif ninapro=='db7':
    subNum=21
  elif ninapro=='bio':
    subNum=10
  elif ninapro=='dba':
    subNum=18
  elif ninapro=='dbb-1':
    subNum=10
  elif ninapro=='dbb-all':
    subNum=20
  elif ninapro=='dbb-all-session':
    subNum=10
  elif ninapro=='femg':
    subNum=28
  for i in range(subNum):
    geindex = str(i).rjust(3, '0')
    files='data/pretrain/'+ninapro+'/'+mode+'_'+geindex+'.tfrecords'
    fileList.append(files)
  return fileList

def tfrecords_reader_dataset(fileList,shuffle_buffer_size,batch_size,ninapro,epoch, n_readers=1,
                             n_parse_threads=5
                             ):
  dataset = tf.data.Dataset.list_files(fileList)
  dataset = dataset.interleave(
    lambda filename: tf.data.TFRecordDataset(
      filename),
    cycle_length=n_readers
  )
  dataset = dataset.map(lambda x:parse_example_merge(x,ninapro),
                        num_parallel_calls=n_parse_threads)
  dataset = dataset.shuffle(shuffle_buffer_size,seed=666)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(epoch)
  return dataset

expected_features_merge = {
  "X": tf.io.FixedLenFeature([], dtype=tf.string),
  "Y": tf.io.FixedLenFeature([], dtype=tf.string),
}
def parse_example_merge(serialized_example,ninapro):
  example = tf.io.parse_single_example(serialized_example,
                                       expected_features_merge)
  X = tf.io.decode_raw(example["X"], out_type=tf.float32)
  Y = tf.io.decode_raw(example["Y"], out_type=tf.float32)
  if (ninapro == 'db2')|(ninapro == 'db3'):
    X = tf.reshape(X, [48, 1, 20])
    Y = tf.reshape(Y, [50])
  elif ninapro == 'db7':
    X = tf.reshape(X, [48, 1, 20])
    # X3 = tf.reshape(X3, [72, 1, 28])
    Y = tf.reshape(Y, [41])
  elif (ninapro=='dba')|(ninapro=='dbb-1')|(ninapro=='dbb-all')|(ninapro=='dbb-all-session'):
    X = tf.reshape(X, [150, 128])
    Y = tf.reshape(Y, [8])
  elif ninapro=='femg':
    X = tf.reshape(X, [150, 64])
    Y = tf.reshape(Y, [38])
  elif ninapro=='bio':
    X = tf.reshape(X, [150, 6])
    Y = tf.reshape(Y, [6])

  return {'td_input': X}, {'output_softmax': Y}
if __name__ == '__main__':
  # IBio_session('000')
  ICapgMyo_getGanData('dbb-2','002')
  f = scio.loadmat('data/capgMyo/DBb/002-001-002.mat')
  f2 = scio.loadmat('data/gan/dbb/002-001-002.mat')

  IFemg('femg','001')
  ICapgMyo_dba_odd_even_slide('dbb-all','011')
  data = f['data']
  # f = mat4py.loadmat('data/semg_data/ninapro_db1/000/000/000_000_010.mat')
  label=f['label'][0][0]

  dp='data/semg_data/ninapro_db1'
  subject='001'
  INiaPro_db1_split(dp,subject,True)
  # x = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [11, 12], [13, 14]]
  # x = np.array(x, dtype='float')
  # x=x*0.1
  # data=ICapgMyo_dba_odd_even('data/semg_data/DBa','001')
  print('111')
  # INiaPro_db1()

