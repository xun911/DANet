from turtle import width
from matplotlib.pyplot import axis, sca
from tensorflow.keras.layers import Conv2D,Conv1D,MaxPooling1D, Dense, Flatten, Input, MaxPooling2D, Dropout, LocallyConnected2D, BatchNormalization, ReLU, ZeroPadding2D,Activation,concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD,Adam
import tensorflow as tf
from keras import backend as K
import keras
# flag=keras.backend.image_data_format()
# print(flag)
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, TimeDistributed, Dense, Bidirectional, GRU, Layer
from keras import initializers, optimizers, regularizers, constraints
from tensorflow.keras import layers
def buile_model(input_shape, classes):
    model = Sequential()
    height = input_shape[0]
    width = input_shape[1]
    #------------------------------------------------------#
    #channel, height, width
    #第一层卷积层1, 16, 8 -> 64, 16, 8
    #------------------------------------------------------#
    #BatchNormal层，作为开始层需要输入input_shape，axis=1表示特征层,-1表示所有层
    trainabel = True
    scale=True
    dataformat = 'channels_last',
    model.add(BatchNormalization(input_shape = input_shape, axis=-1, momentum=0.9,trainable=trainabel))
    #model.add(ZeroPadding2D(padding=(1.h5, 1.h5), data_format='channels_first', input_shape = (1.h5, height, width)))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), data_format='channels_last', padding='same'))
    model.add(BatchNormalization(axis=1, momentum=0.9, scale=scale, trainable=trainabel))
    model.add(ReLU())


    #------------------------------------------------------#
    #channel, height, width
    #第二层卷积层64, 16, 8 -> 64, 16, 8
    #------------------------------------------------------#
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), data_format='channels_last',padding='same'))
    model.add(BatchNormalization(axis=1, momentum=0.9, scale=scale, trainable=trainabel))
    model.add(ReLU())
    #------------------------------------------------------#
    #第三层局部卷积层64, 16, 8 -> 64, 16, 8
    #------------------------------------------------------#
    model.add(LocallyConnected2D(filters=64, kernel_size=1, strides=(1, 1), padding='valid', data_format='channels_last'))
    model.add(BatchNormalization(axis=1, momentum=0.9, scale=scale, trainable=trainabel))
    model.add(ReLU())
    #------------------------------------------------------#
    #第四层局部卷积层64, 16, 8 -> 64, 16, 8
    #------------------------------------------------------#
    model.add(LocallyConnected2D(filters=64, kernel_size=1, strides=(1, 1), padding='valid', data_format='channels_last'))
    model.add(BatchNormalization(axis=1, momentum=0.9, scale=scale, trainable=trainabel))
    model.add(ReLU())
    model.add(Dropout(rate=0.5))
    #------------------------------------------------------#
    #第五层全连接层64, 16, 8 -> 512
    #------------------------------------------------------#
    model.add(Flatten(data_format = 'channels_last'))
    model.add(Dense(512))
    model.add(BatchNormalization(axis=1, momentum=0.9, scale=scale, trainable=trainabel))
    model.add(ReLU())
    model.add(Dropout(rate=0.5))
    #------------------------------------------------------#
    #第六层全连接层512 -> 512
    #------------------------------------------------------#
    model.add(Dense(512))
    model.add(BatchNormalization(axis=1, momentum=0.9, scale=scale, trainable=trainabel))
    model.add(ReLU())
    model.add(Dropout(rate=0.5))
    #------------------------------------------------------#
    #第七层全连接层512 -> 128
    #------------------------------------------------------#
    model.add(Dense(128))
    model.add(BatchNormalization(axis=1, momentum=0.9, scale=scale, trainable=trainabel))
    model.add(ReLU())
    #------------------------------------------------------#
    #第八层全连接层128 -> 8
    #------------------------------------------------------#
    model.add(Dense(classes, activation='softmax', name = 'dense_last'))

    return model

seq_len = 150
cellNeurons = 512
dropout = 0.5
l2=0.0
denseNeurons = 512
def model_2SRNN_my(input_shape, classes):
    model = Sequential()
    height = input_shape[0]
    width = input_shape[1]
    #------------------------------------------------------#


    model.add(TimeDistributed(Dense(128,
                                    kernel_initializer='identity',
                                    bias_initializer='zeros',
                                    name='customNn',
                                    activation=None), input_shape=(seq_len, width), name='td',
                              trainable=False))

    # model.add(Conv1D(64,3,activation='relu'))
    # model.add(BatchNormalization(axis=1, momentum=0.9))
    # model.add(ReLU())

    model.add(GRU(cellNeurons, recurrent_dropout=dropout, name='rnn', trainable=True, return_sequences=True,
                   kernel_regularizer=regularizers.l2(l2)))
    model.add(GRU(cellNeurons, recurrent_dropout=dropout, name='rnn_2nd_layer', trainable=True,
                   kernel_regularizer=regularizers.l2(l2)))
    # model.add(LSTM(cellNeurons, recurrent_dropout=dropout, name='rnn', trainable=True, return_sequences=True,
    #               kernel_regularizer=regularizers.l2(l2)))
    # model.add(LSTM(cellNeurons, recurrent_dropout=dropout, name='rnn_2nd_layer', trainable=True,
    #               kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(denseNeurons, name='nn', trainable=True, kernel_regularizer=regularizers.l2(l2)))
    model.add(Dropout(dropout, name='nn_dropout', trainable=True))
    model.add(Dense(classes, activation="softmax", name='output_softmax', trainable=True,
                    kernel_regularizer=regularizers.l2(l2)))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.adam_v2.Adam(learning_rate=0.001, decay=0.0001),
                  metrics=["accuracy"])

    return model

def model_2SRNN(input_shape, classes):
    model = Sequential()
    height = input_shape[0]
    width = input_shape[1]
    #------------------------------------------------------#


    model.add(TimeDistributed(Dense(128,
                                    kernel_initializer='identity',
                                    bias_initializer='zeros',
                                    name='customNn',
                                    activation=None), input_shape=(seq_len, width), name='td',
                              trainable=False))
    model.add(LSTM(cellNeurons, recurrent_dropout=dropout, name='rnn', trainable=True, return_sequences=True,
                   kernel_regularizer=regularizers.l2(l2)))
    model.add(LSTM(cellNeurons, recurrent_dropout=dropout, name='rnn_2nd_layer', trainable=True,
                   kernel_regularizer=regularizers.l2(l2)))
    model.add(Dense(denseNeurons, name='nn', trainable=True, kernel_regularizer=regularizers.l2(l2)))
    model.add(Dropout(dropout, name='nn_dropout', trainable=True))
    model.add(Dense(classes, activation="softmax", name='output_softmax', trainable=True,
                    kernel_regularizer=regularizers.l2(l2)))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.adam_v2.Adam(learning_rate=0.001, decay=0.0001),
                  metrics=["accuracy"])

    return model

class DomainAdaptationLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(DomainAdaptationLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        source_features, target_features = inputs
        alpha = 0.1  # 超参数，控制源域和目标域之间的知识转移
        return source_features + alpha * (target_features - source_features)




def model_2SRNN_adapt(input_shape, classes,dataset):
    if dataset=='dbb-2':
        # dataset='dbb-all-session'
        dataset = 'dbb-1'
    souceModel = load_model('model/pretrain/'+dataset+'.h5')
    for layer in souceModel.layers:
        layer.trainable = False
    baseModel_1=souceModel.layers[0]
    baseModel_1._name="7777"
    targetModel = keras.Sequential([
        baseModel_1,
        souceModel.layers[1],
        souceModel.layers[2],
        # layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(classes, activation="softmax",name='output_softmax', trainable=True,
                    kernel_regularizer=regularizers.l2(l2))  # 根据任务定义合适的输出层
    ])
    # targetModel=Dense(512)(targetModel.output)
    # targetModel=Dropout(0.5)(targetModel)
    # targetModel=Dense(classes, activation="softmax", name='output_softmax', trainable=True,
    #                 kernel_regularizer=regularizers.l2(l2))(targetModel)





    adaptation_layer = DomainAdaptationLayer()
    source_features = souceModel.get_layer('rnn_2nd_layer').output
    target_features = targetModel.layers[2].output
    adapted_features = adaptation_layer([source_features, target_features])
    output = targetModel.layers[-1](adapted_features)
    adapted_model = keras.Model(inputs=[souceModel.input, targetModel.input], outputs=output)

    # multiFineTuneModel = toMultiGpuModel(fineTuneModel)
    adapted_model.compile(loss="categorical_crossentropy",
                          optimizer=optimizers.adam_v2.Adam(learning_rate=0.001, decay=0.0001),
                          metrics=["accuracy"])
    # fineTuneModel.compile(loss="sparse_categorical_crossentropy",
    #                       optimizer=optimizers.adam_v2.Adam(learning_rate=0.001, decay=0.0001),
    #                       metrics=["accuracy"])

    # Test optimizer's state:
    # print(fineTuneModel.optimizer.get_config())
    # print(dir(fineTuneModel.optimizer))
    # print(fineTuneModel.optimizer.lr)

    return adapted_model

def model_2SRNN_Pretrain(input_shape, classes,dataset):
    if dataset=='dbb-2':
        # dataset='dbb-all-session'
        dataset = 'dbb-1'
    fineTuneModel = load_model('model/pretrain/'+dataset+'.h5')
    fineTuneModel.get_layer('td').trainable = True
    adaptationVersion=1
    if adaptationVersion == 2:
        fineTuneModel.get_layer('td').activation = 'relu'
    fineTuneModel.get_layer('rnn').trainable = False
    if fineTuneModel.get_layer('rnn_2nd_layer') != None:
        fineTuneModel.get_layer('rnn_2nd_layer').trainable = False
    fineTuneModel.get_layer('nn').trainable = False
    fineTuneModel.get_layer('nn_dropout').trainable = False
    fineTuneModel.get_layer('output_softmax').trainable = False

    if adaptationVersion == 3:
        fineTuneModel.get_layer('td').activation = 'relu'
        fineTuneModel.name = "existingModel"
        newModel = Sequential()
        newModel.add(TimeDistributed(Dense(128,
                                           kernel_initializer='identity',
                                           bias_initializer='zeros',
                                           activation='relu'), input_shape=(seq_len, 128), name='td0',
                                     trainable=True))
        newModel.add(fineTuneModel)
        fineTuneModel = newModel
    if adaptationVersion == 4:  # initializer does not work with this initializer cause it is not square
        fineTuneModel.get_layer('td').activation = 'relu'
        fineTuneModel.name = "existingModel"
        newModel = Sequential()
        newModel.add(TimeDistributed(Dense(10 * 128,
                                           kernel_initializer='identity',
                                           bias_initializer='zeros',
                                           activation='relu'), input_shape=(seq_len, 128), name='td0',
                                     trainable=True))
        newModel.add(fineTuneModel)
        fineTuneModel = newModel


    # multiFineTuneModel = toMultiGpuModel(fineTuneModel)
    fineTuneModel.compile(loss="categorical_crossentropy",
                          optimizer=optimizers.adam_v2.Adam(learning_rate=0.001, decay=0.0001),
                          metrics=["accuracy"])
    # fineTuneModel.compile(loss="sparse_categorical_crossentropy",
    #                       optimizer=optimizers.adam_v2.Adam(learning_rate=0.001, decay=0.0001),
    #                       metrics=["accuracy"])

    # Test optimizer's state:
    # print(fineTuneModel.optimizer.get_config())
    # print(dir(fineTuneModel.optimizer))
    # print(fineTuneModel.optimizer.lr)

    return fineTuneModel

'''卷积层
参数：输入，flag（为True即第一个卷积层需要在前面增加一个batch处理）
'''
def getConv(input,flag):
    if flag:
        batch = BatchNormalization(axis=-1, momentum=0.9)(input)
        #取消偏置
        conv = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',use_bias=True)(batch)
    else:
        conv = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',use_bias=True)(input)
    batch = BatchNormalization(axis=3, momentum=0.9)(conv)
    out = ReLU()(batch)
    return out
'''局部卷积层
参数：输入，flag（为True即最后一个卷积层需要在后面增加一个drop处理）
'''
def getLC(input,flag):
    lc = LocallyConnected2D(filters=64, kernel_size=1, strides=(1, 1), padding='valid',implementation=1,data_format='channels_last')(input)
    # lc = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', use_bias=False)(input)
    # lc = Dense(64)(input)
    # lc = get_smooth_pixel_reduce(input)
    batch = BatchNormalization(axis=3, momentum=0.9)(lc)
    rel = ReLU()(batch)
    if flag:
        drop=Dropout(rate=0.5)(rel)
        return drop
    return rel
'''全连接层
参数：输入，flag（为True即第一个全连接层需要在前面增加一个flatten处理）
'''
def getFC(input,flag):
    if flag:
        fla = Flatten()(input)
        den=Dense(512,use_bias=True)(fla)
    else:
        den = Dense(512,use_bias=True)(input)
    batch = BatchNormalization(axis=-1, momentum=0.9)(den)
    rel = ReLU()(batch)
    drop = Dropout(rate=0.5)(rel)
    return drop
'''全连接层2
参数：输入
'''
def getFC2(input):
    den = Dense(52,use_bias=True)(input)
    batch = BatchNormalization(axis=-1, momentum=0.9)(den)
    rel = ReLU()(batch)
    return rel

def buile_model2(input_shape, classes):
    # model = Sequential()
    height = input_shape[0]
    width = input_shape[1]
    inputx = Input(input_shape, name='inputx0')
    # ------------------------------------------------------#
    # channel, height, width
    # 第一层卷积层1, 16, 8 -> 64, 16, 8
    # ------------------------------------------------------#
    # BatchNormal层，作为开始层需要输入input_shape，axis=1表示特征层,-1表示所有层
    conv1 = getConv(inputx, True)
    # ------------------------------------------------------#
    # channel, height, width
    # 第二层卷积层64, 16, 8 -> 64, 16, 8
    # ------------------------------------------------------#
    conv2= getConv(conv1, False)
    # ------------------------------------------------------#
    # 第三层局部卷积层64, 16, 8 -> 64, 16, 8
    # ------------------------------------------------------#
    lc1=getLC(conv2,False)
    # ------------------------------------------------------#
    # 第四层局部卷积层64, 16, 8 -> 64, 16, 8
    # ------------------------------------------------------#
    lc2=getLC(lc1,True)
    # ------------------------------------------------------#
    # 第五层全连接层64, 16, 8 -> 512
    # ------------------------------------------------------#
    fc1=getFC(lc2,True)
    # ------------------------------------------------------#
    # 第六层全连接层512 -> 512
    # ------------------------------------------------------#
    fc2 = getFC(fc1, False)
    # ------------------------------------------------------#
    # 第七层全连接层512 -> 128
    # ------------------------------------------------------#
    fc3=getFC2(fc2)
    # ------------------------------------------------------#
    # 第八层全连接层128 -> 8
    # ------------------------------------------------------#
    # 第八层 softmax ->52
    den= Dense(classes, activation='softmax', name='output')(fc3)

    model= Model(inputs=inputx, outputs=den)

    # 定义优化器
    opt = SGD(lr=0.1, decay=0.0001)
    #
    # 模型编译
    model.compile(loss='CategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model_2SRNN_adapt("11","1",'dbb-2')

