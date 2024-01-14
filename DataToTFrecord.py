import gc

from tqdm import tqdm
import tensorflow as tf

from dataloader import ICapgMyo_dba_odd_even_slide,ICapgMyo_dba_odd_even_slide_session,IFemg,IBio_session
import numpy as np

def DataToTFrecord_losocv(dataset,subject):
    trainCountDict = {}
    testCountDict = {}

    if dataset == 'dba':
        allList = ['001', '002', '003','004', '005','006', '007','008',
                       '009','010', '011','012',  '013','014','015','016','017', '018']
        testList=[subject]
        trainList = set(allList)-set(testList)
        dataPath = 'data/capgMyo/DBb'

    else:
        print('please check data path')
        return
    for i in tqdm(range(len(trainList))):
        print('the {0} ------------'.format(i))
        trainPath = 'data/pretrain/' + dataset + '/train_' + str(i).rjust(3, '0') + '.tfrecords'
        writer_train = tf.io.TFRecordWriter(trainPath)
        if dataset == 'dba':
            preData = ICapgMyo_dba_odd_even_slide_session(dataset,trainList[i])
        for j in range(preData[0].shape[0]):
            example_train = tf.train.Example(features=tf.train.Features(feature={
                "X": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[preData[0][j].astype(np.float32).tostring()])),
                "Y": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[preData[1][j].astype(np.float32).tostring()]))
            }))
            writer_train.write(example_train.SerializeToString())
        writer_train.close()
        trainCountDict[trainList[i]] = preData[0].shape[0]
        del preData
    for i in tqdm(range(len(testList))):
        print('the {0} ------------'.format(i))
        testPath = 'data/pretrain/' + dataset + '/test_' + str(i).rjust(3, '0') + '.tfrecords'
        writer_test = tf.io.TFRecordWriter(testPath)
        if dataset == 'dba':
            preData = ICapgMyo_dba_odd_even_slide_session(dataset, testList[i])
        for j in range(preData[0].shape[0]):
            example_train = tf.train.Example(features=tf.train.Features(feature={
                "X": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[preData[0][j].astype(np.float32).tostring()])),
                "Y": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[preData[1][j].astype(np.float32).tostring()]))
            }))
            writer_test.write(example_train.SerializeToString())
            # 序列化为字符串
        writer_test.close()
        testCountDict[trainList[i]] = preData[0].shape[0]

        del preData
        gc.collect()
    np.save('data/pretrain/' + dataset + '/trainCount.npy', trainCountDict)
    np.save('data/pretrain/' + dataset + '/testCount.npy', testCountDict)
    # del preData4

    # dataCountToDict(ninapro)
    # dataCountToDict(ninapro)
    print('-------------------sucess!-------------------')


def DataToTFrecord_session(ninapro):
    trainCountDict = {}
    testCountDict = {}

    if ninapro == 'dbb-all-session':
        trainList = ['001',  '003', '005', '007',
                       '009', '011',  '013',  '015',  '017', '019']
        testList = ['002', '004',  '006',  '008',
                      '010', '012', '014',  '016',  '018','020']
        dataPath = 'data/capgMyo/DBb'

    else:
        print('please check data path')
        return
    for i in tqdm(range(len(trainList))):
        print('the {0} ------------'.format(i))
        trainPath = 'data/pretrain/' + ninapro + '/train_' + str(i).rjust(3, '0') + '.tfrecords'
        writer_train = tf.io.TFRecordWriter(trainPath)
        preData = ICapgMyo_dba_odd_even_slide_session(ninapro,trainList[i])
        for j in range(preData[0].shape[0]):
            example_train = tf.train.Example(features=tf.train.Features(feature={
                "X": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[preData[0][j].astype(np.float32).tostring()])),
                "Y": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[preData[1][j].astype(np.float32).tostring()]))
            }))
            writer_train.write(example_train.SerializeToString())
        writer_train.close()
        trainCountDict[trainList[i]] = preData[0].shape[0]
        del preData
    for i in tqdm(range(len(testList))):
        print('the {0} ------------'.format(i))
        testPath = 'data/pretrain/' + ninapro + '/test_' + str(i).rjust(3, '0') + '.tfrecords'
        writer_test = tf.io.TFRecordWriter(testPath)
        preData = ICapgMyo_dba_odd_even_slide_session(ninapro, testList[i])
        for j in range(preData[0].shape[0]):
            example_train = tf.train.Example(features=tf.train.Features(feature={
                "X": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[preData[0][j].astype(np.float32).tostring()])),
                "Y": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[preData[1][j].astype(np.float32).tostring()]))
            }))
            writer_test.write(example_train.SerializeToString())
            # 序列化为字符串
        writer_test.close()
        testCountDict[trainList[i]] = preData[0].shape[0]

        del preData
        gc.collect()
    np.save('data/pretrain/' + ninapro + '/trainCount.npy', trainCountDict)
    np.save('data/pretrain/' + ninapro + '/testCount.npy', testCountDict)
    # del preData4

    # dataCountToDict(ninapro)
    # dataCountToDict(ninapro)
    print('-------------------sucess!-------------------')




def DataToTFrecord_single(ds):
  # subjectList = ['000']
  # ninapro=data_path.split('_')[1]
  trainCountDict = {}
  testCountDict = {}

  if ds == 'dba':
      subjectList = ['001', '002', '003', '004', '005', '006', '007', '008',
                     '009', '010', '011', '012', '013', '014', '015', '016', '017', '018']
      dataPath = 'data/capgMyo/DBa'
  elif ds=='dbb-1':
    subjectList = ['001', '003', '005', '007',
                   '009', '011', '013', '015', '017', '019']
    dataPath = 'data/capgMyo/DBb'
  elif ds=='dbb-all':
    subjectList = ['001', '002','003','004', '005','006', '007',
                  '008', '009','010', '011','012', '013','014', '015','016', '017','018', '019','020']
    dataPath = 'data/capgMyo/DBb'
  elif ds=='femg':
    subjectList = [ '001', '002', '003','004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                   '014', '015', '016', '017', '018', '019', '021','022','023','024','025','026','027','028']
  elif ds=='bio':
    subjectList = ['000','001', '002', '003','004', '005', '006', '007', '008', '009']
  else:
      print('please check data path')
      return
  for i in tqdm(range(len(subjectList))):
    print('the {0} ------------'.format(i))
    trainPath='data/pretrain/'+ds+'/train_'+subjectList[i]+'.tfrecords'
    testPath = 'data/pretrain/'+ds+'/test_' + subjectList[i] + '.tfrecords'
    writer_train = tf.io.TFRecordWriter(trainPath)
    writer_test = tf.io.TFRecordWriter(testPath)

    # preData1 = dataloader.INiaPro_db1_feature_split_single(data_path, subjectList[i], 'psr', True,ninapro)
    if ds=='femg':
        preData = IFemg(ds, subjectList[i])
    elif ds=='bio':
        preData = IBio_session(subjectList[i])
    else:
        preData = ICapgMyo_dba_odd_even_slide(ds, subjectList[i])

    for j in range(preData[0].shape[0]):
      example_train = tf.train.Example(features=tf.train.Features(feature={
        "X": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData[0][j].astype(np.float32).tostring()])),
        "Y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData[2][j].astype(np.float32).tostring()]))
      }))
      writer_train.write(example_train.SerializeToString())  # 序列化为字符串
    for k in range(preData[1].shape[0]):
      example_test = tf.train.Example(features=tf.train.Features(feature={
        "X": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData[1][k].astype(np.float32).tostring()])),
        "Y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData[3][k].astype(np.float32).tostring()]))
      }))
      writer_test.write(example_test.SerializeToString())  # 序列化为字符串
    writer_train.close()
    writer_test.close()
    trainCountDict[subjectList[i]] = preData[0].shape[0]
    testCountDict[subjectList[i]] = preData[1].shape[0]

    del preData
    gc.collect()
  np.save('data/pretrain/' + ds + '/trainCount.npy', trainCountDict)
  np.save('data/pretrain/' + ds + '/testCount.npy', testCountDict)
    # del preData4

  # dataCountToDict(ninapro)
  # dataCountToDict(ninapro)
  print('-------------------sucess!-------------------')


def countTfRecord(filepath):
    count = 0
    for record in tf.compat.v1.io.tf_record_iterator(filepath):
      count += 1
    print('数据{0}的数量是{1}'.format(filepath, count))
    return count

if __name__ == '__main__':
    # DataToTFrecord_single('../../extract_features/out_features/ninapro-db5-var-raw-prepro-lowpass-win-40-stride-20','db5')
    # x1=countTfRecord('../../data/pretrain/db1/train_000.tfrecords')
    # DataToTFrecord_losocv('dba','001')
    DataToTFrecord_single('bio')
    # DataToTFrecord_single('femg')
    # DataToTFrecord_session('dbb-all-session')
    # DataToTFrecord_single('dbb-all')
