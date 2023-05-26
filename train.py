# -*- coding: utf-8 -*-
'''
    @Time    : 23/05/2023, 9:04 PM
    @Author  : cj
'''
import argparse
import os

import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam

from util import *
from model import *


def train(config):

    # ------ check GPU ------ #
    os.environ['CUDA_VISIBLE_DEVICE'] = config.GPUNUM
    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.per_process_gpu_memory_fraction = 0.95
    gpuconfig.allow_soft_palcement = True
    gpuconfig.log_device_placement = False
    sess = tf.Session(config=gpuconfig)
    KTF.set_session(sess)
    tf.global_variables_initializer().run(session=sess)

    # ------ random seed ------ #
    np.random.seed(21)

    # ------ data loader ------ #
    train_dataread = data_read(np.arange(0, 86), pathname=config.data_read_path)
    print('1')
    valid_dataread = data_read(np.arange(86, 94), pathname=config.data_val_path)
    print('2')

    train_datagenerate = data_generate(data=train_dataread,
                                       batch_size=config.batch_size,
                                       input_data_patch_shape=config.input_patch_shape,
                                       output_data_patch_shape=config.output_patch_shape,
                                       output_symbol=config.output_symbol)
    valid_datagenerate = data_generate(data=valid_dataread,
                                       batch_size=config.batch_size,
                                       input_data_patch_shape=config.input_patch_shape,
                                       output_data_patch_shape=config.output_patch_shape,
                                       output_symbol=config.output_symbol)

    # ------ build model ------ #
    model = config.model_name(config.input_patch_shape, config.output_patch_shape)
    model.compile(optimizer=Adam(lr=config.learing_rate), loss='mse')

    model_input_path = '../models/{}'.format(config.output_metrics_name)
    model_check_point = ModelCheckpoint(filepath='{}/model_{}.hdf5'.format(model_input_path, '1'),
                                       verbose=1,
                                       monitor='val_loss',
                                       save_best_only=True)
    tensor_board = TensorBoard(log_dir='../logs/{}/{}'.format(config.output_metrics_name, '1'))
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    print('ok! ')
    model.fit_genertator(train_datagenerate,
                         steps_per_epoch=200,
                         epochs=config.epochs,
                         validation_data=valid_datagenerate,
                         validation_steps=100,
                         callbacls=[model_check_point, tensor_board, early_stop],
                         verbose=1)
    model.save('{}/model_me_{}.hdf5'.format(model_input_path, '1'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment info
    parser.add_argument('--model_name', type=str, default='vnet')
    parser.add_argument('--output_metrics_name', type=str, default='vnet')
    parser.add_argument('--data_read_path', type=str, default='./data/data_read/')
    parser.add_argument('--data_val_path', type=str, default='./data/data_val')
    parser.add_argument('--GPU_NUM', type=str, default='0')

    # dataset parameters
    parser.add_argument('--input_patch_shape', type=list, default=[64, 64, 64])
    parser.add_argument('--output_patch_shape', type=list, default=[32, 32, 32])
    parser.add_argument('--output_symbol', type=str, default='qsm_images')
    parser.add_argument('--batch_size', type=int, default=8)

    # model hyper-parameters

    # training hyper-parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)

    # misc

    config = parser.parse_args()
    train(config)
