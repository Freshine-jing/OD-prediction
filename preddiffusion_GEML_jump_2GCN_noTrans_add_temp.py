import sys
import shutil
import os
import time
from datetime import datetime
import scipy.sparse as ss
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Activation, Flatten, Dense, Reshape, Concatenate, Add, Lambda, Layer, add, multiply, \
    TimeDistributed, UpSampling2D, concatenate, BatchNormalization, LSTM, Dot
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K

from load_data import *
from metric import *

###########################Reproducible#############################
import random

np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

tf.set_random_seed(100)


###################################################################

class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    def __init__(self, units, support=1,
                 activation=None, ## 激活函数
                 use_bias=False, ## 偏置项
                 kernel_initializer='glorot_uniform',  # Gaussian distribution
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def build(self, input_shapes): ## 定义kernel 卷积核函数
        features_shape = input_shapes[0]
        # assert len(features_shape) == 2
        input_dim = features_shape[-1]

        ## 确定卷积核结构 定义全连接
        self.kernel = self.add_weight(shape=(input_dim, self.units), ## units可改
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(GraphConvolution, self).build(input_shapes)

    ## 类似forward 定义执行流程
    def call(self, inputs, mask=None):
        features = inputs[0]
        links = inputs[1]

        result = K.batch_dot(links, features, axes=[2, 1])
        output = K.dot(result, self.kernel) ## 做图卷积

        if self.bias:
            output += self.bias

        return self.activation(output)

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


def sequence_GCN(input_seq, adj_seq, unit, act='relu', **kwargs):
    GCN = GraphConvolution(unit, activation=act, **kwargs) ## 使用GCN 直接调用call
    embed = []
    for n in range(input_seq.shape[1]):
        frame = Lambda(lambda x: x[:, n, :, :])(input_seq) ## 把时序数据分成很多片段
        adj = Lambda(lambda x: x[:, n, :, :])(adj_seq)
        embed.append(GCN([frame, adj])) ## 对每一帧数据做处理后放进embed中
    output = Lambda(lambda x: tf.stack(x, axis=1))(embed) ##
    return output


def getModel():
    X_fea = Input(batch_shape=(BATCHSIZE, TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH))
    X_temp = Input(batch_shape=(BATCHSIZE, day_fea))
    X_adj = Input(batch_shape=(BATCHSIZE, TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH))
    X_seman = Input(batch_shape=(BATCHSIZE, TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH))

    x1_nebh = sequence_GCN(X_fea, X_adj, ENCODER_DIM)
    x2_nebh = sequence_GCN(x1_nebh, X_adj, ENCODER_DIM)

    x1_seman = sequence_GCN(X_fea, X_seman, ENCODER_DIM)
    x2_seman = sequence_GCN(x1_seman, X_seman, ENCODER_DIM)

    dens1 = Dense(units=10, activation='relu')(X_temp) ## 内置全连接接口
    dens2 = Dense(units=TIMESTEP * HEIGHT * WIDTH * ENCODER_DIM, activation='relu')(dens1) ## 可以多加几个隐藏层
    hmeta = Reshape((TIMESTEP, HEIGHT * WIDTH, ENCODER_DIM))(dens2)

    embed_fea = add([x2_nebh, x2_seman, hmeta])
    embed_fea = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(embed_fea) ##维度重排列
    # lstm_fea = TimeDistributed(LSTM(2 * ENCODER_DIM, return_sequences=False))(embed_fea)
    embed_fea = Lambda(lambda x: K.reshape(x, (BATCHSIZE * HEIGHT * WIDTH, TIMESTEP, ENCODER_DIM)))(embed_fea)
    lstm_fea = LSTM(ENCODER_DIM, return_sequences=False)(embed_fea)
    out = Lambda(lambda x: K.reshape(x, (BATCHSIZE, HEIGHT * WIDTH, ENCODER_DIM)))(lstm_fea)

    model = Model(inputs=[X_fea, X_temp, X_adj, X_seman], outputs=out)
    return model


def testModel(name, testData, dayinfo):
    print('Model Evaluation Started ...', time.ctime())

    assert os.path.exists(PATH + '/' + name + '.h5'), 'model is not existing'
    model = getModel()
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.load_weights(PATH + '/' + name + '.h5')
    model.summary()

    test_gene = test_generator_GEML(testData, dayinfo, batch=1, step=step, jump=True)
    test_step = (testData.shape[0] - start_GEML) // BATCHSIZE
    testY = get_true_GEML(testData, step=step, jump=True)
    print('testY shape: {}'.format(testY.shape))

    pred = model.predict_generator(test_gene, steps=test_step, verbose=1)
    print('pred shape: {}'.format(pred.shape))
    pred_sparse = ss.csr_matrix(pred.reshape(pred.shape[0], -1))
    re_pred_sparse, re_testY = pred_sparse * MAX_DIFFUSION, testY * MAX_DIFFUSION
    mse_score = mse_func(re_testY, re_pred_sparse)
    rmse_score = rmse_func(re_testY, re_pred_sparse)
    mae_score = mae_func(re_testY, re_pred_sparse)
    mape_score = mape_func(re_testY, re_pred_sparse)

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Rescaled MSE on testData, {}\n".format(mse_score))
    f.write("Rescaled RMSE on testData, {}\n".format(rmse_score))
    f.write("Rescaled MAE on testData, {}\n".format(mae_score))
    f.write("Rescaled MAPE on testData, {:.3f}%\n".format(100 * mape_score))
    f.close()

    print('*' * 40)
    print('MSE', mse_score)
    print('RMSE', rmse_score)
    print('MAE', mae_score)
    print('MAPE {:.3f}%'.format(100 * mape_score))
    print('Model Evaluation Ended ...', time.ctime())

    predictionDiffu = re_pred_sparse
    groundtruthDiffu = re_testY
    ss.save_npz(PATH + '/' + MODELNAME + '_prediction.npz', predictionDiffu)
    ss.save_npz(PATH + '/' + MODELNAME + '_groundtruth.npz', groundtruthDiffu)


def trainModel(name, trainData, dayinfo):
    print('Model Training Started ...', time.ctime())

    train_num = int(trainData.shape[0] * (1 - SPLIT)) ## 训练数据量 传入的一段时序数据
    print('train num: {}, val num: {}'.format(train_num, trainData.shape[0] * SPLIT))
    train_gene = data_generator_GEML(trainData[:train_num], dayinfo[:train_num], BATCHSIZE, step=step, jump=True)
    val_gene = data_generator_GEML(trainData[train_num - start_GEML:], dayinfo[train_num - start_GEML:], BATCHSIZE,
                                   step=step, jump=True) ## 训练过程中判断当前训练的结果是否过拟合
    train_step = (train_num - TIMESTEP) // BATCHSIZE ## 维稳
    val_step = (trainData.shape[0] * SPLIT) // BATCHSIZE

    model = getModel()
    model.compile(loss=LOSS, optimizer=OPTIMIZER) ## 定义好的loss函数和优化器（做梯度下降）
    model.summary() ## 做log
    csv_logger = CSVLogger(PATH + '/' + name + '.log')
    checkpointer = ModelCheckpoint(filepath=PATH + '/' + name + '.h5', verbose=1, save_best_only=True) ## 保存训练好的model
    LR = LearningRateScheduler(lambda epoch: LEARN) ## 学习率，随着训练轮数进行衰减的曲率，调成参数更新的幅度
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto') ## 防止过拟合
    model.fit_generator(train_gene, steps_per_epoch=train_step, epochs=EPOCH,
                        validation_data=val_gene, validation_steps=val_step,
                        callbacks=[csv_logger, checkpointer, LR, early_stopping]) ## 塞进超参数

    pred = model.predict_generator(test_generator_GEML(trainData[train_num - start_GEML:],
                                                       dayinfo[train_num - start_GEML:],
                                                       BATCHSIZE, step=step, jump=True), steps=val_step) ## 得到预测结果
    pred = pred.reshape((-1, HEIGHT * WIDTH, HEIGHT, WIDTH))
    print('pred shape: {}'.format(pred.shape))
    pred_sparse = ss.csr_matrix(pred.reshape(pred.shape[0], -1))
    valY = get_true_GEML(trainData[train_num - start_GEML:], step=step, jump=True)

    re_pred_sparse, re_valY = pred_sparse * MAX_DIFFUSION, valY * MAX_DIFFUSION
    mse_score = mse_func(re_valY, re_pred_sparse)
    rmse_score = rmse_func(re_valY, re_pred_sparse)
    mae_score = mae_func(re_valY, re_pred_sparse)
    mape_score = mape_func(re_valY, re_pred_sparse)

    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("Rescaled MSE on valData, {}\n".format(mse_score))
    f.write("Rescaled RMSE on valData, {}\n".format(rmse_score))
    f.write("Rescaled MAE on valData, {}\n".format(mae_score))
    f.write("Rescaled MAPE on valData, {:.3f}%\n".format(100 * mape_score))
    f.close()

    print('*' * 40)
    print('MSE', mse_score)
    print('RMSE', rmse_score)
    print('MAE', mae_score)
    print('MAPE {:.3f}%'.format(100 * mape_score))
    print('Model Train Ended ...', time.ctime())


################# Parameter Setting #######################
MODELNAME = 'GEML_jump_2GCN_noTrans_add_temp'

# KEYWORD = 'preddiffusion_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M%S")
# KEYWORD = 'preddiffusion_GEML_200108020249'
# PATH = '../' + KEYWORD


################# Parameter Setting #######################


if __name__ == '__main__':
    param = sys.argv
    if len(param) == 3:
        GPU = param[-1]
    else:
        GPU = '0'
    config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = GPU
    set_session(tf.Session(graph=tf.get_default_graph(), config=config)) ## 一个session执行一个训练

    step = int(param[-2]) ##
    KEYWORD = 'preddiffusion_' + MODELNAME + '_' + str(step) + '_' + datetime.now().strftime("%y%m%d%H%M%S")
    # KEYWORD = 'preddiffusion_GEML_jump_2GCN_noTrans_add_temp_0_200131023056'
    PATH = '../' + KEYWORD

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('load_data.py', PATH)
    shutil.copy2('metric.py', PATH)

    diffusion_data = ss.load_npz(diffusionFile) ## load data
    diffusion_data = diffusion_data / MAX_DIFFUSION
    dayinfo = np.genfromtxt(dayinfoFile, delimiter=',', skip_header=1)
    print('data.shape, dayinfo.shape', diffusion_data.shape, dayinfo.shape)
    train_Num = int(diffusion_data.shape[0] * trainRatio)

    print(KEYWORD, 'training started', time.ctime())
    trainvalidateData = diffusion_data[:train_Num]
    trainvalidateDay = dayinfo[:train_Num, ]
    print('trainvalidateData.shape', trainvalidateData.shape)
    trainModel(MODELNAME, trainvalidateData, trainvalidateDay)

    print(KEYWORD, 'testing started', time.ctime())
    testData = diffusion_data[train_Num - start_GEML:]
    testDay = dayinfo[train_Num - start_GEML:]
    print('testData.shape', testData.shape)
    testModel(MODELNAME, testData, testDay)
