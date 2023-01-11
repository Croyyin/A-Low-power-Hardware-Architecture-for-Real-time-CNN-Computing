from curses import A_CHARTEXT
import os
import re
import time
from matplotlib.pyplot import delaxes
import numpy as np
import pickle
import glob
import random
import matplotlib.pyplot as plt
from spatial_test import data_clp
from experiment import setup_seed, stft_general_data, stft_general_test
from train import train
from utils.dataTrans import get_dataloader
from sklearn.model_selection import train_test_split

# audio
import librosa
import librosa.display
import sklearn
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from model import Unquantified_2layer_CNN, Unquantified_2layer_CNN_sp, Unquantified_2layer_CNN_sp_test, VGG16_base

# 参数
in_channels_list = [2, 1]
out_channels_list =[1, 1]
kernel_size = [3, 3]
stride = [1, 1]
pooling_stride_o = (2, 1)
pooling_stride_n = (2, 2)
feature_map_size = (1025, 173)
BATCH_SIZE = 64




# 计算并保存运行结果
def infrc():

    # 数据准备
    data_x_list, data_y = data_clp()
    data_x_list = np.array(data_x_list)[:,0:2,:,:]
    data_y = np.array(data_y)[0:2]

    # 模型准备
    model_n = Unquantified_2layer_CNN_sp_test(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_n, feature_map_size)
    model_n.load_state_dict(torch.load('models/very/stft/noverlap/1/best_model7266.pt'))

    model_o = Unquantified_2layer_CNN_sp_test(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_o, feature_map_size)
    model_o.load_state_dict(torch.load('models/very/stft/overlap/1/best_model7266.pt'))

    model_n.test_parameter_set()
    model_o.test_parameter_set()
    
    # 统计相关参数
    mode = 1
    column_o = None
    column_n = None
    result_o = None
    result_n = None

    count = 0
    for data_x_n in tqdm(data_x_list, desc='time'):
        data_x = torch.from_numpy(data_x_n)
        # overlapping pooling 预测
        predict_o, result_o, column_o = model_o.fc_forward(data_x, result_o, column_o, mode)
        
        # 与上述相同
        predict_n, result_n, column_n = model_n.fc_forward(data_x, result_n, column_n, mode)

        if mode == 1:
            mode = 0

        count += 1

        if count == 5:
            break

if __name__ == '__main__':
    infrc()
    
    