
# 开发时间 2022/4/16 15:34
import os
import numpy as np
import pynvml
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
from scipy import stats
import xlwt
import shutil
import pymrmr
import  pandas as pd

from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import seaborn as sns
from collections import Counter

#返回一串1-10通道重排列之后的序号
def genIndex(chanums):

      index = []
      i = 1
      j = i+1

      if (chanums % 2) == 0:
         Ns = chanums+1
      else:
         Ns = chanums


      index.append(1)
      t = chr(i+ord('A'))
      while(i!=j):
          l = ""
          l = l+chr(i+ord('A'))
          l = l+chr(j+ord('A'))
          r = ""
          r = r+chr(j+ord('A'))
          r = r+chr(i+ord('A'))
          if(j>Ns):
              j = 1
          elif(t.find(l)==-1 and t.find(r)==-1):
              index.append(j)
              t = t+chr(j+ord('A'))
              i = j
              j = i+1
          else:
              j = j+1



      new_index = []
      if (chanums % 2) == 0:
          for i in range(len(index)):
              if index[i] != chanums+1:
                 new_index.append(index[i])
          index = new_index

      index = np.array(index)
      index = index-1
      return index
#获得通道重新排列的数据
def get_sig_img(data,sigmig_index):
    res=[]
    for sample in data:
        signal_img = sample[sigmig_index]
        signal_img = signal_img[:-1]
        # signal_img=np.array(signal_img)
        res.append(signal_img)
    res=np.array(res)
    # res1 =res.reshape(-1,sigmig_index.shape[0]-1)
    return res
def get_sig_img2(data, sigimg_index):
#     ch_num = data.shape[0]
#     sigimg_index = genIndex(ch_num)
     signal_img = data[sigimg_index]
     signal_img = signal_img[:-1]
#     print signal_img.shape
     return signal_img
#获取指定GPU剩余显存
def getGpuMrmory(id):
    pynvml.nvmlInit()
    # 这里的0是GPU id
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('GPU{0}剩余{1}M'.format(id,meminfo.free/1024**2))



def CheckFolder(dataPath):
    flag=os.path.exists(dataPath)
    if flag!=True:
        os.makedirs(dataPath)
''' 该函数实现窗口宽度为七、滑动步长为1的滑动窗口截取序列数据 
参数：数据，窗口宽度，窗口步长，窗口起始值
'''
def sliding_window(data, sw_width, sw_step, in_start=0):
    # data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))  # 将以周为单位的样本展平为以天为单位的序列
    X, y = [], []

    for i in range(len(data)):
        in_end = in_start + sw_width
        # out_end = in_end + n_out
        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；不然丢弃该样本
        if in_end <= len(data):
            # 训练数据以滑动步长1截取
            train_seq = data[in_start:in_end]

            X.append(train_seq)

        in_start += sw_step

    return np.array(X)



def get_segments(data, window, stride):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window-stride)* data.shape[1]
    )
def get_segments_image(data, window, stride):
    chnum = data.shape[1];
    data=windowed_view(
        data.flat,
        window * data.shape[1],
        (window-stride)* data.shape[1]
    )
    data= data.reshape(-1, window, chnum)
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


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y
def butter_filter(data,wLow,wHigh,fs,order,zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt
    nyq = 0.5 * fs
    # cut = cut / nyq
    high=wHigh/nyq
    low=wLow/nyq
    b, a = butter(order, [low,high],'bandpass')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cut, fs, order, zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = butter(order, cut, btype='low')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y

def _get_sigimg_aux(data, sigimg_index):
    return np.transpose(get_sig_img(data.T, sigimg_index))

def get_sigimg1(data, sigimg_index):
        res = []
        for sample in data:
            amp=_get_sigimg_aux(sample, sigimg_index)
            res.append(amp[np.newaxis, ...])


        res = np.concatenate(res, axis=0)
#        res = res.reshape(res.shape[0], 1, res.shape[1], res.shape[2])
        res = res.reshape(res.shape[0], res.shape[1], res.shape[2], -1)
        return res
def downsample(data, step):
    return data[::step].copy()

def SaveMat(outputPath,data,subject,gesture,trial):
    #创建层级文件夹
    outputDir = os.path.join(
        outputPath,
        '{0:03d}',
        '{1:03d}').format(subject,gesture)
    if os.path.isdir(outputDir) is False:
        os.makedirs(outputDir)
    #保存mat文件
    out_path = os.path.join(
        outputDir,
        '{0:03d}_{1:03d}_{2:03d}.mat').format(subject, gesture,trial)
    scio.savemat(out_path,
                 {'data': data, 'label': gesture, 'subject': subject, 'trial': trial})
    print("Subject %d Gesture %d Trial %d saved!" % (subject, gesture, trial))
def showIMU(true,pre):

    # true_1=true[:,0:1]
    # true_2=true[:,1:2]
    # true_3 = true[:, 2:3]
    # pre_1 = pre[:, 0:1]
    # pre_2 = pre[:, 1:2]
    # pre_3 = pre[:, 2:3]
    # true_1=true[0][0:300]
    # true_2=true[1][0:300]
    # true_3 = true[2][0:300]
    # true_4 = true[3][0:300]
    # pre_1 = pre[0][0:300]
    # pre_2 = pre[1][0:300]
    # pre_3 = pre[2][0:300]
    # pre_4 = pre[3][0:300]
    true_1 = true[0][300:700]
    true_2 = true[1][300:700]
    true_3 = true[2][300:700]
    true_4 = true[3][300:700]
    pre_1 = pre[0][300:700]
    pre_2 = pre[1][300:700]
    pre_3 = pre[2][300:700]
    pre_4 = pre[3][300:700]
    plt.figure(figsize=(20,6))
    ax1=plt.subplot(411)
    ax1.plot(true_1,color='blue',label='true')
    ax1.plot(pre_1,color='red',label='predict')
    ax1.legend(loc=1)
    ax2 = plt.subplot(412)
    ax2.plot(true_2, color='blue',label='true')
    ax2.plot(pre_2, color='red',label='predict')
    ax2.legend(loc=1)
    ax3 = plt.subplot(413)
    ax3.plot(true_3, color='blue',label='true')
    ax3.plot(pre_3, color='red',label='predict')
    ax3.legend(loc=1)
    ax4 = plt.subplot(414)
    ax4.plot(true_4, color='blue', label='true')
    ax4.plot(pre_4, color='red', label='predict')
    ax4.legend(loc=1)
    plt.show()
    print('111')
def showIMU3(data,ges):


    plt.figure(figsize=(3,3))
    filename='test'+str(ges)+'.svg'

    plt.plot(data,color='#006EBD')
    plt.savefig(filename, bbox_inches='tight')


    # plt.axis('off')
    # plt
    plt.show()
    print('111')
def showIMU2(true,pre):

    true_1 = true[0:300]

    pre_1 = pre[0:300]

    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(211)
    ax1.plot(true_1,color='blue',label='true')
    ax1.plot(pre_1, color='red', label='predict')
    ax1.legend(loc=1)

    ax2= plt.subplot(212)
    ax2.plot(pre_1, color='red', label='predict')
    ax2.legend(loc=1)
    plt.show()
    print('111')
def min_max(data,min,max):
    scaler=MinMaxScaler(feature_range=(min,max),copy=True)
    scaler.fit(data)
    norm_value=scaler.transform(data)
    return norm_value
def standSc(data):
    scaler = StandardScaler()
    # scaler.fit(data)
    data = scaler.fit_transform(data.T).T
    # for col in data.columns:
    #     data[col]=scaler.fit_transform(data[col].values.reshape(-1,1))
    return data
def loss_fft(y_true, y_pred):
    fft_true = np.abs(np.fft.rfft(y_true))
    fft_pred = np.abs(np.fft.rfft(y_pred))
    loss = np.mean(np.square(np.subtract(fft_true,fft_pred)))
    return loss, fft_true, fft_pred

def cross_correlation(y_true, y_pred):
    cc = np.correlate(y_true,y_pred)
    return cc

def dtw_distance(y_true, y_pred):
    #distance = dtw.distance(y_true, y_pred)
    distance, path = fastdtw(y_true, y_pred, dist=euclidean)
    return distance
def save_excel(dataList,nameList,savePath):
    # 例如我们要存储两个list：name_list 和 err_list 到 Excel 两列
    # dataList = [[10, 20, 30],[0.99, 0.98, 0.97]]  # 示例数据
    # nameList = ['one', 'two', 'three']  # 示例数据
    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet1.write(0, 0, "序号")
    for i in range(len(nameList)):
        sheet1.write(0, i+1, nameList[i])  # 第1行第1列
        # sheet1.write(0, 1, "数量")  # 第1行第2列
        # sheet1.write(0, 2, "误差")  # 第1行第3列

    # 循环填入数据
    for i in range(len(dataList[0])):
        sheet1.write(i + 1, 0, i) # 第1列序号
        for j in range(len(dataList)):
            sheet1.write(i + 1, j+1, dataList[j][i])
            # sheet1.write(i + 1, 1, name_list[i])  # 第2列数量
        # sheet1.write(i + 1, 2, err_list[i])  # 第3列误差
    file.save(savePath)
    print('training record are saved in '+savePath)
def butter_filter(data,wLow,wHigh,fs,order,zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt
    nyq = 0.5 * fs
    # cut = cut / nyq
    # high = wHigh
    # low = wLow
    high=wHigh/nyq
    low=wLow/nyq
    b, a = butter(order, [low,high],'bandpass')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y
def emgPlot(emgData,emgData2):
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.figure(figsize=(20, 8))


    # label = dfraw[:, 12]

    # dfraw=np.column_stack(dfraw)
    # for i in range(len(dfraw)):
    #     if (dfraw[i, 11] != 0):
    #         dfraw[i, 12] = 0.00060
    ax1 = plt.subplot(211)
    ax1.plot(emgData[:, 0], label='label', linestyle='-')
    ax1.plot(emgData[:, 1], label='rep', linestyle='-')
    ax1.plot(emgData[:, 2], label='ch3', linestyle='-')
    ax1.plot(emgData[:, 3], label='ch4', linestyle='-')
    ax1.plot(emgData[:, 4], label='ch5', linestyle='-')
    ax1.plot(emgData[:, 5], label='ch6', linestyle='-')
    ax1.plot(emgData[:, 6], label='ch7', linestyle='-')
    ax1.plot(emgData[:, 7], label='ch8', linestyle='-')
    ax1.plot(emgData[:, 8], label='ch9', linestyle='-')
    ax1.plot(emgData[:, 9], label='ch10', linestyle='-')
    ax1.plot(emgData[:, 10], label='ch11', linestyle='-')
    ax1.plot(emgData[:, 11], label='ch12', linestyle='-')
    plt.ticklabel_format(style='sci',axis='y', scilimits=(0, 0))
    plt.ylabel('sEMG signal',size='13')
    plt.legend(loc=1)
    ax2 = plt.subplot(212)
    ax2.plot(emgData2[:, 0], label='label', linestyle='-')
    ax2.plot(emgData2[:, 1], label='rep', linestyle='-')
    ax2.plot(emgData2[:, 2], label='ch3', linestyle='-')
    ax2.plot(emgData2[:, 3], label='ch4', linestyle='-')
    ax2.plot(emgData2[:, 4], label='ch5', linestyle='-')
    ax2.plot(emgData2[:, 5], label='ch6', linestyle='-')
    ax2.plot(emgData2[:, 6], label='ch7', linestyle='-')
    ax2.plot(emgData2[:, 7], label='ch8', linestyle='-')
    ax2.plot(emgData2[:, 8], label='ch9', linestyle='-')
    ax2.plot(emgData2[:, 9], label='ch10', linestyle='-')
    ax2.plot(emgData[:, 10], label='ch11', linestyle='-')
    ax2.plot(emgData[:, 11], label='ch12', linestyle='-')
    # plt.plot(dfraw[:, 12],color='c', linestyle='--',label='Action')

    plt.xlabel('Samples',size='13')
    plt.ylabel('sEMG signal',size='13')
    plt.legend(loc=1)
    plt.show()

    #     plt.savefig(r"E:\PaperSupport\SVG\action分割.svg",format="svg")
def emgPlot2(emgData):
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.figure(figsize=(6, 4))


    # label = dfraw[:, 12]

    # dfraw=np.column_stack(dfraw)
    # for i in range(len(dfraw)):
    #     if (dfraw[i, 11] != 0):
    #         dfraw[i, 12] = 0.00060
    ax1 = plt.subplot(111)
    ax1.plot(emgData[:, 0], label='label', linestyle='-',linewidth=2.0 )
    ax1.plot(emgData[:, 1], label='rep', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 2], label='ch3', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 3], label='ch4', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 4], label='ch5', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 5], label='ch6', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 6], label='ch7', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 7], label='ch8', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 8], label='ch9', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 9], label='ch10', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 10], label='ch11', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 11], label='ch12', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 12], label='ch12', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 13], label='ch12', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 14], label='ch12', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 15], label='ch12', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 16], label='ch12', linestyle='-',linewidth=2.0)
    ax1.plot(emgData[:, 17], label='ch12', linestyle='-',linewidth=2.0)
    # ax1.plot(emgData[:, 18], label='ch12', linestyle='-')
    # ax1.plot(emgData[:, 19], label='ch12', linestyle='-')
    # ax1.plot(emgData[:, 20], label='ch12', linestyle='-')
    # ax1.plot(emgData[:, 21], label='ch12', linestyle='-')
    # ax1.plot(emgData[:, 22], label='ch12', linestyle='-')

    plt.ticklabel_format(style='sci',axis='y', scilimits=(0, 0))
    plt.ylabel('sEMG signal',size='13')
    plt.xlabel('Samples',size='13')
    plt.ylabel('sEMG signal',size='13')
    # plt.legend(loc=1)
    # plt.show()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("imu.svg",format="svg",bbox_inches='tight',pad_inches=0)
def emg_mav(signal):
    signal_abs = [abs(s) for s in signal]
    signal_abs = np.array(signal_abs)
    if len(signal_abs) == 0:
        return 0
    else:
        return np.mean(signal_abs)
def extract_emg_feature2(x, feature_name):

    res = []
    for i in range(x.shape[0]):
        func = 'emg_'+feature_name
        res.append(eval(str(func))(x[i,:]))
    res =np.vstack(res)
    return res
def imuToMav(data):
    feature = [np.transpose(extract_emg_feature2(seg.T, 'mav')) for seg in data]
    feature = np.array(feature)
    return feature

def mRmR():
    # 生成数据

    imu=scio.loadmat('/data/ywt/emg-ges2/data/gan/db2/035/002/035_002_000.mat')['data'][200]
    imu=np.reshape(imu,[imu.shape[2],imu.shape[0]])
    imu = pd.DataFrame(columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                                                          '13','14','15','16','17','18','19','20','21','22','23','24',
                                                          '25','26','27','28','29','30','31','32','33','34','35','36'],data=imu)

    label = [35] * 20
    label=np.reshape(label,[20,1])

    label2=pd.DataFrame(columns=['1'],data=label)
    res=pd.concat([label2,imu],axis=1,join='inner')
    # imu=pd.DataFrame(columns=['1','2','3','4','5','6','7','8','9','10','11','12',
    #                           '13','14','15','16','17','18','19','20','21','22','23','24',
    #                           '25','26','27','28','29','30','31','32','33','34','35','36'],data=imu)
    #
    # imu['0']=label
    # print("数据集样本数: ", data.shape[0])
    # print("数据集特征数: ", data.shape[1])
    mr=pymrmr.mRMR(res,'MIQ',12)
    mr2=mr.sort()
    print('..')

def showMatrix(true,pred):
    cnf_matrix = confusion_matrix(true, pred)
    cnf_matrix=cnf_matrix.astype('float')/cnf_matrix.sum(axis=1)[:,np.newaxis]
    a1=18
    a2=33
    # a1 = 0
    # a2 = 17
    # a1 = 34
    # a2 = 49
    cnf_matrix=cnf_matrix[a1:a2+1,a1:a2+1]
    cnf_matrix=np.around(cnf_matrix,decimals=2)
    classes = list(range(a1,a2+1))
    # 类别标签
    sns.set()
    f, ax = plt.subplots(figsize=(10, 8))
    # sns.heatmap(C2, annot=T, ax=ax,cmap=plt.cm.Blues,vmax=1500,
    #             xticklabels=classes,yticklabels=classes)  # plot heatmap
    sns.heatmap(cnf_matrix, annot=True, ax=ax, cmap=plt.cm.Blues, vmax=1,
                xticklabels=classes,yticklabels=classes)  # plot heatmap
    # ax.set_title('gesture {0}-{1}\n\n'.format(a1,a2),pad=1);
    ax.set_xlabel('Predicted Label',fontweight='bold')
    ax.set_ylabel('True label',fontweight='bold')  #
    # plt.show()
    sns.despine()
    plt.savefig("model_emg+imu/confusion_matrix.svg",bbox_inches='tight', format="svg")

    plt.show()
def GetFCScore(true,pred):
    # C2 = confusion_matrix(true, pred)
    # FP = sum(C2.sum(axis=0)) - sum(np.diag(C2))  # 假正样本数
    # FN = sum(C2.sum(axis=1)) - sum(np.diag(C2))  # 假负样本数
    # TP = sum(np.diag(C2))  # 真正样本数
    # TN = sum(C2.sum().flatten()) - (FP + FN + TP)  # 真负样本数
    # SUM = TP + FP
    # acc=(TP+TN)/(TP+FP+TN+FN)
    # precision = TP / (TP + FP)  # 查准率，又名准确率
    # recall = TP / (TP + FN)  # 查全率，又名召回率
    # f1=(2*precision*recall)/(precision+recall)
    acc=accuracy_score(true,pred)
    precision=precision_score(true,pred,average='macro')
    recall=recall_score(true,pred,average='macro')
    f1=f1_score(true,pred,average='macro')
    return acc,precision,recall,f1


'''获得数组中出现次数最多数字
'''
def Find_Majority(array):
    #找出数组中元素出现次数并按从大到小排序
    collection_words = Counter(array)
    #找出出现最多次数的
    most_counterNum = collection_words.most_common(1)
    mostwords=most_counterNum[0][0]
    return mostwords

if __name__ == '__main__':

    # mRmR()
    # x1=np.prod((20,36,1))
    # index=genIndex(36)
    # sc=scio.loadmat('001_001_000.mat')['data']
    # sc2=scio.loadmat('/data/ywt/emg-ges2/data/ninapro_db2/001/001/001_001_000.mat')['data']
    # sc3=np.abs(sc2)
    # true=[0,0,0,0,0,1,1,1,2,2,2,3,3,3,0,0]
    # pre=[0,1,1,2,2,0,1,1,1,1,2,2,3,3,1,1]
    # num=Find_Majority(pre)
    # showMatrix(true,pre)
    emg=scio.loadmat('/data/ywt/emg-ges2/rawdata/ninapro_db7/s1/S1_E1_A1.mat')['acc'].astype(np.float32)[0:3000,:]
    emg2=emg[:,[20,22,23,24,25,26,27,28,29,32,34,35]]
    # emg=abs(emg)
    # data = np.transpose([butter_filter(ch, 20, 450, 2000, 10, zero_phase=True) for ch in emg.T])
    # data = np.transpose([butter_lowpass_filter(ch, 500, 2000, 1, zero_phase=True) for ch in emg.T])
    emgPlot2(emg)
    # shutil.rmtree('/data/ywt/emg-ges2/model/gan/db7')
    # os.mkdir('/data/ywt/emg-ges2/model/gan/db7')


    imu=downsample(imu,10)
    imu=get_segments_image(imu,20,1)


    imu=imuToMav(imu)
    emg2 = scio.loadmat('/data/ywt/emg-ges2/extract_features/out_features/ninapro-db2-imu-lowpass-win-20-stride-1/000/000/000_000_000_mav.mat')['data']
    showIMU3(emg1,16)

    # emg1=emg_mav(emg)
    mav=[]
    for i in range(emg.shape[0]):
        s=emg_mav(emg[i])
        mav.append(s)
    mav=np.array(mav)



    # imu = downsample(imu, step=20)
    # imu = get_segments_image(imu, 20, 1)
    col1=imu[:,0:1]
    col1=np.reshape(col1,(col1.shape[0]))
    xxx=col1/mav
    col2=imu[:,2:3]
    col3=col1/col2
    col3=np.divide(col1,col2,col3)

    xx = downsample(xx, step=20)
    xx = get_segments_image(xx, 20, 1)
    mav=scio.loadmat('/data/ywt/emg-ges2/extract_features/out_features/ninapro-db2-downsample20-var-raw-prepro-lowpass-win-20-stride-1/000/001/000_001_000_mav.mat')['data']
    sti1=scio.loadmat('/data/ywt/emg-ges2/rawdata/ninapro_db5/s1/S1_E1_A1.mat')['stimulus'][0:20000]
    rep1 = scio.loadmat('/data/ywt/emg-ges2/rawdata/ninapro_db5/s1/S1_E1_A1.mat')['repetition'][0:20000]
    data1=np.hstack((sti1,rep1))
    sti2 = scio.loadmat('/data/ywt/emg-ges2/rawdata/ninapro_db5/s1/S1_E1_A1.mat')['restimulus'][0:20000]
    rep2 = scio.loadmat('/data/ywt/emg-ges2/rawdata/ninapro_db5/s1/S1_E1_A1.mat')['rerepetition'][0:20000]
    data2= np.hstack((sti2, rep2))
    # emgData = scio.loadmat('/data/ywt/emg-ges2/rawdata/ninapro_db2/s1/S1_E1_A1.mat')['restimulus']

    # emgData3 = np.transpose([butter_lowpass_filter(ch, 0.5, 2000, 1, zero_phase=True) for ch in xx.T])
    # emgData3 = np.transpose([butter_filter(ch, 20, 450, 2000, 1, zero_phase=True) for ch in xx.T])
    emgPlot(data1, data2)

    genList=[]
    gen=scio.loadmat('data/gan/db2/000/000/000_000_000.mat')['data']
    # real = scio.loadmat('data/imu/db2/000/001/000_001_000.mat')['data']
    for i in range(gen.shape[0]):
        genList.append(gen[i].flatten())
    genList=np.array(genList)
    realList = []
    real=scio.loadmat('data/imu/db2/000/001/000_001_000.mat')['data']
    real = downsample(real, step=20)
    real = get_segments_image(real, 20, 1)
    real = np.expand_dims(real, axis=3)
    real = real.transpose(0, 2, 3, 1)
    for i in range(real.shape[0]):
        realList.append(real[i].flatten())
    realList = np.array(realList)
    showIMU(realList, genList)

    # real2= np.transpose([butter_lowpass_filter(ch, 1, 2000, 1, zero_phase=True) for ch in real.T])
    emgPlot(genList,realList)

    # dataPath1='/data/ywt/emg-ges2/extract_features/out_features/ninapro-db2-downsample20-var-raw-prepro-lowpass-win-20-stride-1/003/000/003_000_000_arc.mat'
    dataPath2 = '/data/ywt/emg-ges2/data/ninapro_db2/010/001/010_001_000.mat'
    str = 'filter'
    dataList = [[10, 20, 30], [0.99, 0.98, 0.97]]
    nameList = ['one', 'two', 'three']
    save_excel(dataList,nameList,'output/db2/test.xls')

    data11 = get_segments(data, 20, 10)
    data11 = data11.reshape(-1, 20, 12)

    data22 = downsample(data, step=20)
    data22 = get_segments(data22, 20, 1)
    data22 = data22.reshape(-1, 20, 12)
    data3 = scio.loadmat(dataPath3)['data']

    imu = scio.loadmat(dataPath1)['data']
    imu = downsample(imu, step=20)
    imu = get_segments_image(imu, 20, 1)
    genimu=scio.loadmat('data/gan/db2/000/001/000_001_000.mat')['data']
    emg = scio.loadmat(dataPath2)['data']
    raw=scio.loadmat(data3)
    imu=imu.flatten()[0:800]

    fft_metric,fft_ref,fft_gen=loss_fft(imu,imu)
    dtw_metric=dtw_distance(imu,imu)
    cc_metric=cross_correlation(imu,imu)[0]
    r=stats.pearsonr(imu,imu)
    r2=pearsonr(imu,imu)
    cc=list(pearsonr(imu,imu))[0]
    emg = min_max(emg, 0, 1.0)



    window=40
    stride=20
    chnum1 = emg.shape[1];
    chnum2 = imu.shape[1];
    data1 = get_segments(emg, window, stride)
    data1 = data1.reshape(-1, window, chnum1)
    data1=data1*255

    data2= get_segments(imu, window, stride)
    data2 = data2.reshape(-1, window, chnum2)
    data2 = data2 * 255
    # data = np.expand_dims(data, axis=3)
    # x1=data1[0]
    # x2=data2[1]
    # plt.imshow(x1)
    # plt.show()
    # plt.imshow(x2)
    # plt.imshow(out.astype('uint8'))
    # plt.show()
    # showIMU(imu,emg)
    # dataPath='model/'+'DBa/'
    # CheckFolder(dataPath)
    # x=genIndex(10)
    # sigmig_index = genIndex(10)
    f = scio.loadmat(
     'extract_features/out_features/ninapro-db5-var-raw-prepro-lowpass-win-40-stride-20/000/001/000_001_000_dwt.mat')
    data = f['data']
    # ress=get_sig_img2(data,sigmig_index)
    # ress = np.expand_dims(ress, axis=2)
    #
    # ress22=get_sigimg1(f['data'],sigmig_index)[0]
    # res=get_sigimg1(data,sigmig_index)
    # getGpuMrmory(0)
    print('1111')
