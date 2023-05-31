# -*- coding: utf-8 -*-
'''
    @Time    : 26/05/2023, 10:17 AM
    @Author  : cj
'''
import argparse
import os

import numpy as np
from scipy.io import savemat

from util import data_read, data_predict
import net_model


def test(config):
    # ------ check GPU ------ #
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM

    net_func = getattr(net_model, config.model_name)
    model = net_func(config.unwrap_data_patch_shape, config.output_patch_shape, 1)
    model.load_weights(config.model_path)

    test_data = data_read(config.test_data_order, config.test_data_path)

    if not os.path.exists(config.output_data_path):
        os.makedirs(config.output_data_path)

    for j, data_symbol in enumerate(test_data):
        print('This is for ', j+28)
        unwrap_data = data_symbol[config.symbol]

        data_symbol['autoQSM_images'], data_symbol['patch_images'] = data_predict(model, unwrap_data,
                                                                                  config.unwrap_data_patch_shape,
                                                                                  config.output_patch_shape)
        savemat('{}/subject_{}.mat'.format(config.output_data_path, config.test_data_order[j]), data_symbol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model info
    parser.add_argument('--model_name', type=str, default='vnet')
    parser.add_argument('--model_path', type=str, default='./models/vnet/model_me_1.hdf5')  # model_final_1.h5df
    parser.add_argument('--GPU_NUM', type=str, default='2')

    # data path
    parser.add_argument('--test_data_order', type=list, default=np.arange(28, 50))  # stop at 28, ori=(0, 50)
    parser.add_argument('--test_data_path', type=str, default='/DATA2023/cj/AutoQSM/data/data_test/')

    # dataset parameters
    parser.add_argument('--unwrap_data_patch_shape', type=list, default=[64, 64, 64])
    parser.add_argument('--output_patch_shape', type=list, default=[32, 32, 32])
    parser.add_argument('--symbol', type=str, default='unwrap_images')

    # output path
    parser.add_argument('--output_data_path', type=str, default='./test_result')

    config = parser.parse_args()
    test(config)
