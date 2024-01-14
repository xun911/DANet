# 开发时间 2022/8/10 17:14
from __future__ import print_function, division
import tensorflow as tf
# from keras.datasets import mnist
# from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import LeakyReLU, LSTM, concatenate, Lambda
from tensorflow.keras.layers import UpSampling2D, Conv2D,Conv1D,Conv2DTranspose,Conv1DTranspose,Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
import numpy as np
# import dataloader
import scipy.io as scio
from utils import showIMU,showIMU2
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler
from scipy import stats
from imutils import paths
from statistics import mean
from utils import get_segments_image,SaveMat2,dtw_distance,downsample
from dataloader import IGAN_emg_single,IGAN_emg_All
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,r2_score
import time
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras import initializers, regularizers, constraints
import shutil
from math import sqrt

# np.random.seed(666)
def Set_GPU():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_list:
        #设置显存不占满
        tf.config.experimental.set_memory_growth(gpu, True)
        #设置显存占用最大值
        tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]
        )
class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.

    # Example

    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)

        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)

        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```

    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.

    # Input shape
        2D tensor with shape: `(samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.

    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class ACGAN():
    def __init__(self,ninapro,subject):
        # Input shape

        # self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 8
        # self.latent_dim = 100
        self.ninapro=ninapro
        self.subject=subject

        optimizer = Adam(0.0002)
        # optimizer = SGD(0.001)
        losses = ['binary_crossentropy']
        # losses = ['mse', 'categorical_crossentropy']
        # Build the generator
        self.generator = self.build_generator()
        # self.generator.summary()
        inputx0 = Input([1, 128,1], name='inputx0')
        # label = Input(shape=())  # 标签
        img= self.generator([inputx0])
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # self.discriminator.summary()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image

        valid= self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(inputx0,valid)
        self.combined.compile(loss=losses,
                              optimizer=optimizer,
                                   metrics=['accuracy'])


    '''卷积层
    参数：输入，flag（为True即第一个卷积层需要在前面增加一个batch处理）
    '''

    def getConv(self,input,filter):

        # conv = Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
        conv = Conv1D(filters=filter, kernel_size=4, strides=2, padding='same')(input)
        batch = BatchNormalization(axis=-1, momentum=0.9)(conv)
        # output = LeakyReLU()(batch)
        output = Activation('relu')(batch)
        return output
    def getConvTranspose(self,input, filter):

        # conv = Conv2DTranspose(filters=filter, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
        conv = Conv2DTranspose(filters=filter, kernel_size=3, strides=(1,2), padding='same')(input)
        batch = BatchNormalization(axis=-1, momentum=0.9)(conv)
        # output = LeakyReLU()(batch)
        output = Activation('relu')(batch)
        return output
    def build_generator(self):
        inputx0 = Input([1, 128,1], name='inputx0')
        output=Flatten()(inputx0)
        # input = tf.reshape(inputx0, [-1, 1,128,1])
        output=Dense(128)(output)
        output = tf.reshape(output, [-1, 128, 1, 1])
        output = Conv2DTranspose(filters=64, kernel_size=3, strides=(2,1), padding='same')(output)
        output = BatchNormalization(axis=-1, momentum=0.9)(output)
        output = LeakyReLU(alpha=0.2)(output)

        output = Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 1), padding='same')(output)
        output = BatchNormalization(axis=-1, momentum=0.9)(output)
        output = LeakyReLU(alpha=0.2)(output)

        output = Conv2DTranspose(filters=1, kernel_size=3, strides=(2, 1), padding='same')(output)






        # output = self.getConvTranspose(input, 128)
        # output = self.getConvTranspose(output, 64)
        # output = self.getConvTranspose(output, 32)
        # output = self.getConvTranspose(output, 16)
        # output = self.getConv(output, 16)

        # output = self.getConv(output, 32)
        # output = self.getConv(output, 64)
        # output = self.getConv(output, 1)
        # output = Dropout(0.2)(output)
        # output = self.getConv(output, 512)
        #
        # output = self.getConvTranspose(output, 512)
        # output = self.getConvTranspose(output, 256)
        # output = self.getConvTranspose(output, 128)
        # output = self.getConvTranspose(output, 64)
        # output = self.getConvTranspose(output, 1)
        # output = Dropout(0.2)(output)

        # output=self.getConvTranspose(inputx0,1)
        output = Flatten()(output)
        output=Dense(128)(output)
        # output = BatchNormalization(axis=-1, momentum=0.9)(output)
        output = Activation('tanh')(output)
        output=tf.reshape(output,[-1,1,128,1])
        res = Model(inputs=[inputx0], outputs=[output])
        return res

    def build_discriminator(self):
        img = Input([1,128,1], name='img')
        # img = tf.reshape(img, [-1, 1,128, 1])
        img1 = tf.reshape(img, [-1, 128, 1, 1])
        output = Conv2D(filters=32, kernel_size=3, strides=(2,1), padding='same')(img1)
        output = LeakyReLU()(output)

        output = Conv2D(filters=64, kernel_size=3, strides=(2, 1), padding='same')(output)
        output = LeakyReLU()(output)
        output = BatchNormalization(axis=-1, momentum=0.9)(output)

        output = Conv2D(filters=128, kernel_size=3, strides=(2, 1), padding='same')(output)
        output = LeakyReLU()(output)
        output = BatchNormalization(axis=-1, momentum=0.9)(output)
        output=Flatten()(output)
        # fft=Lambda(tf.signal.rfft)(flat)
        # fft_abs=Lambda(K.abs)(fft)
        # fft_abs=tf.reshape(fft_abs,[-1,19,1])
        # output1 = Conv1D(filters=16, kernel_size=3, strides=2, padding='same')(fft_abs)
        # output1 = BatchNormalization(axis=-1, momentum=0.9)(output1)
        # output1 = LeakyReLU()(output1)
        # output1 = Dropout(rate=0.2)(output1)
        # output1 = Conv1D(filters=32, kernel_size=3, strides=2, padding='same')(output1)
        # output1 = BatchNormalization(axis=-1, momentum=0.9)(output1)
        # output1 = LeakyReLU()(output1)
        # output1 = Dropout(rate=0.2)(output1)
        # output1 = Conv1D(filters=64, kernel_size=3, strides=2, padding='same')(output1)
        # output1 = BatchNormalization(axis=-1, momentum=0.9)(output1)
        # output1 = LeakyReLU()(output1)
        # output1 = Dropout(rate=0.2)(output1)
        # output1 = Conv1D(filters=64, kernel_size=3, strides=2, padding='same')(output1)
        # output1 = BatchNormalization(axis=-1, momentum=0.9)(output1)
        # output1 = LeakyReLU()(output1)
        # output1 = Dropout(rate=0.2)(output1)
        # output1 = Flatten()(output1)


        # output = Conv1D(filters=16, kernel_size=3, strides=2, padding='same')(img1)
        # output = BatchNormalization(axis=-1, momentum=0.9)(output)
        # output = LeakyReLU()(output)
        # output = Dropout(rate=0.2)(output)
        #
        # output = Conv1D(filters=32, kernel_size=3, strides=2, padding='same')(output)
        # output = BatchNormalization(axis=-1, momentum=0.9)(output)
        # output = LeakyReLU()(output)
        # output = Dropout(rate=0.2)(output)
        #
        # output = Conv1D(filters=64, kernel_size=3, strides=2, padding='same')(output)
        # output = BatchNormalization(axis=-1, momentum=0.9)(output)
        # output = LeakyReLU()(output)
        # output = Dropout(rate=0.2)(output)
        #
        # output = Conv1D(filters=32, kernel_size=3, strides=2, padding='same')(output)
        # output = BatchNormalization(axis=-1, momentum=0.9)(output)
        # output = LeakyReLU()(output)
        # output = Dropout(rate=0.2)(output)
        #
        # output = Conv1D(filters=16, kernel_size=3, strides=2, padding='same')(output)
        # output = BatchNormalization(axis=-1, momentum=0.9)(output)
        # output = LeakyReLU()(output)
        # output = Dropout(rate=0.2)(output)
        # output = Flatten()(output)
        # mini_disc = MinibatchDiscrimination(10, 3)(flat)
        # output=concatenate([output,output1,mini_disc])
        # output = concatenate([output])
        validity = Dense(1,activation='sigmoid')(output)

        discriminator = Model(img, validity)
        return discriminator

    def scheduler(self,models,epoch):
        # 每隔100个epoch，学习率减小为原来的1/2
        if epoch % 500 == 0 and epoch != 0:
            for model in models:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
    def train(self, epochs,originData,targetData,trainLabel, batch_size, sample_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, originData.shape[0], batch_size)
            originData_idx = originData[idx]
            targetData_idx = targetData[idx]
            # trainLabel_idx=trainLabel[idx]
            originData_idx = np.reshape(originData_idx, [originData_idx.shape[0], 1, originData_idx.shape[1],1])
            targetData_idx=np.reshape(targetData_idx,[targetData_idx.shape[0],1,targetData_idx.shape[1],1])

            # Generate a half batch of new images
            gen_imgs= self.generator.predict([originData_idx])




            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(targetData_idx, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch(originData_idx, valid)

            # Plot the progress
            # print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]"
            #       "[DTW Metric: %f][CC Metric: %f]" % (
            #     epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0],dtw_metric,cc_metric[0]))
            if epoch % 100==0:
                generated=gen_imgs.flatten()
                reference=targetData_idx.flatten()
                # dtw_metric=dtw_distance(reference,generated)
                cc_metric=pearsonr(reference,generated)
                rmse=sqrt(mean_squared_error(reference,generated))
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                      "[CC Metric: %f]" "[rmse: %f]"% (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss[0],cc_metric[0],rmse))
                # print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]"% (
                #       epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:

                self.save_model2(epoch)
                # if epoch==0:
                #     self.save_model(epoch,g_loss)
                # else:
                #     self.save_model_select(epoch,g_loss,cc_metric[0],'g_loss')
                # self.sample_images(epoch,trainData,imuData)

    def sample_images(self, epoch,trainData,imuData):

        emgData, imuData, trainLabel = IGAN_emg_imu('000', 'db2', 1)
        gen_imgs = self.generator.predict([emgData,trainLabel])
        # gen_imgs = self.combined.predict(trainData)
        # gen_imgs=gen_imgs.flatten()
        # imuData==imuData.flatten()
        gen_imgs=np.reshape(gen_imgs, (gen_imgs.shape[0], -1))
        print('batch size:',gen_imgs.shape[0])
        true_imgs = np.reshape(imuData, (imuData.shape[0], -1))
        rList=[]
        for i in range(gen_imgs.shape[0]):
            r=stats.pearsonr(gen_imgs[i],true_imgs[i])
            rList.append(r[0])
        print(rList)
        print('pearson average:',mean(rList))
        showIMU(true_imgs,gen_imgs)

        print('11')


    def save_model(self,epoch,num):
        self.generator.save('model/gan/{0}/generator_{1}_e{2}_{3}'.format(self.ninapro,self.subject,epoch,num), save_format="h5")

    def save_model2(self, epoch):
        self.generator.save('model/gan/{0}/{2}/generator_e{1}'.format(self.ninapro,epoch,targetSubject),
                            save_format="h5")
    def save_model_select(self, epoch,g_loss,cc_metric,flag):
        old = os.listdir('model/gan/'+self.ninapro)[0].split('_')[3]
        old=float(old)
        if flag=='g_loss':
            if g_loss<old:
                shutil.rmtree('/data/ywt/emg-ges2/model/gan/'+self.ninapro)
                os.mkdir('model/gan/'+self.ninapro)
                self.generator.save('model/gan/{0}/generator_{1}_e{2}_{3}'.format(self.ninapro,self.subject,epoch,g_loss), save_format="h5")
        else:
            if cc_metric>old:
                shutil.rmtree('/data/ywt/emg-ges2/model/gan/' + self.ninapro)
                os.mkdir('model/gan/' + self.ninapro)
                self.generator.save(
                    'model/gan/{0}/generator_{1}_e{2}_{3}'.format(self.ninapro, self.subject, epoch, cc_metric),
                    save_format="h5")

# subjectList = ['021', '022', '023', '024', '025']


def test(subject,ge,rep):
    genPath = 'model/gan/dbb-2/020/generator_e19500'
    originData, trainLabel = IGAN_emg_single('dbb-1','001', ge,rep)
    targetData, trainLabel =IGAN_emg_single('dbb-2','001',ge,rep)
    # emgData2, imuData2, trainLabel2 = IGAN_emg_imu('db2', '021', 10, '000')
    # imuData0=imuData2[0].flatten('F')
    originData = np.reshape(originData, (originData.shape[0], 1,originData.shape[1]))
    # emgData0=emgData[0].flatten('F')
    generator=tf.keras.models.load_model(genPath)
    gen_imgs = generator.predict(originData)
    # gen_imgs = np.reshape(gen_imgs, (gen_imgs.shape[0], gen_imgs.shape[1],emgData.shape[2]))

    #channel
    # gen_imgs0=gen_imgs[:,:,30:31].flatten()
    # true_imgs0 = imuData[:, 30:31].flatten()

    gen_imgs0=gen_imgs.flatten()
    true_imgs0 = targetData.flatten()
    print('batch size:', gen_imgs.shape[0])
    # true_imgs = np.reshape(imuData, (imuData.shape[0], imuData.shape[1]))
    # true_imgs0=imuData[0].flatten('F')

    pearsoList = []
    rmseList = []
    r2List=[]
    # for i in range(gen_imgs.shape[0]):
    #     pearso = pearsonr(true_imgs[i],gen_imgs[i])[0]
    #     # dtw_metric = dtw_distance(gen_imgs[i], true_imgs[i])
    #     rmse=mean_squared_error(true_imgs[i],gen_imgs[i])
    #     r2=r2_score(true_imgs[i],gen_imgs[i])
    #     pearsoList.append(pearso)
    #     rmseList.append(rmse)
    #     r2List.append(r2)
    # print('{0}pearson average:{1}'.format(genPath, mean(pearsoList)))
    # print('{0}mse average:{1}'.format(genPath, mean(rmseList)))
    # print('{0}r2score average:{1}'.format(genPath, mean(r2List)))
    showIMU2(true_imgs0, gen_imgs0)
    print('111')
    # return mean(pearsoList),mean(rmseList),mean(r2List)



    print('11')

def GenGanImu(dataset,subject):
    if dataset == 'dbb-2':
        classes = 8
        channel = 16
        ge_first_index = 0
    elif dataset=='db3':
        pass
    outPath='data/gan/{0}'.format('dbb')
    # modelPath = 'model/gan/db7/'+os.listdir('model/gan/' + ninapro)[0]
    modelPath='model/gan/{0}/{1}/generator_e6500'.format(dataset,subject)
    # modelPath='model/gan/generator_'+subject+'_e5000'
    generator = tf.keras.models.load_model(modelPath)
    # sigmig_index = genIndex(12)
    testIndex=[2,4,6,8,10]
    for i in range(classes):
        for j in testIndex:
            reIndex = str(j).rjust(3, '0')
            emgData, trainLabel = IGAN_emg_single(dataset,subject, i, reIndex)

            emgData=np.reshape(emgData,[emgData.shape[0],1,emgData.shape[1],1])
            gen_imgs = generator.predict(emgData)
            gen_imgs=np.reshape(gen_imgs,[gen_imgs.shape[0],gen_imgs.shape[2]])

            SaveMat2(outPath, gen_imgs, int(subject), i+1, j)
    # showIMU(true_imgs, gen_imgs)
if __name__ == '__main__':
    Set_GPU()

    #---------------------------------test

    # test('xx','001','001')
    # GenGanImu('dbb-2','002')

    #---------------------------------train
    dataset='dbb-2'
    targetSubject='020'
    start = time.time()
    originList=['001']
    targetList=['001']
    originData,trainLabel = IGAN_emg_single(dataset,'001',-1,-1)
    targetData, trainLabel = IGAN_emg_single(dataset, '001', -1, -1)

    # originData, trainLabel = IGAN_emg_All(dataset, originList)
    # targetData, trainLabel = IGAN_emg_All(dataset, targetList)
    acgan = ACGAN(dataset,targetSubject)
    acgan.train(20000, originData, targetData, trainLabel, batch_size=64, sample_interval=100)
    end = time.time()
    fitTime = end - start
    print("模型训练时长:", fitTime, "s")

    #------------------
    # GenGanImu(dataset, targetSubject)

    print('111')