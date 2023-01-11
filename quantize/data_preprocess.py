from itertools import chain
import os
from matplotlib.pyplot import axis
import numpy as np
import pickle
import glob

from numpy.core.fromnumeric import shape
from pandas.core.indexing import _AtIndexer
from torch.utils import data

import utils.dataTrans as dt

import torch 
from sklearn.model_selection import train_test_split


# wav文件转化为numpy
def wav2np(parent_dir, fold_prename, file_name, save_path_prename):
    # 加载所有数据文件
    data_files = glob.glob(os.path.join(parent_dir, fold_prename + '*', file_name))

    # 数据转换
    data_x, data_y = dt.load_dataset_stft(data_files)
    pickle.dump(data_x, open(save_path_prename + '_x.dat', 'wb'))
    pickle.dump(data_y, open(save_path_prename + '_y.dat', 'wb'))

def wav2np_one_dim(parent_dir, fold_prename, file_name, save_path_prename):
    # 加载所有数据文件
    data_files = glob.glob(os.path.join(parent_dir, fold_prename + '*', file_name))

    # 数据转换
    data_x, data_y = dt.load_dataset_one_dim(data_files)
    pickle.dump(data_x, open(save_path_prename + '_one_dim_x.dat', 'wb'))
    pickle.dump(data_y, open(save_path_prename + '_one_dim_y.dat', 'wb'))

# 从文件读取
def load_data(data_x_path, data_y_path):
    data_x = pickle.load(open(data_x_path, 'rb')).astype(np.float32)
    data_y = pickle.load(open(data_y_path, 'rb')).astype(np.float32)
    return data_x, data_y

# 读取stft通用训练验证集
def stft_general_data():
    # 加载数据
    data_x, data_y = load_data('data/preprocess/stft/n/STFT_train_and_dev_v1_x.dat', 'data/preprocess/stft/n/STFT_train_and_dev_v1_y.dat')
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    return data_x, data_y

# 获取训练验证测试数据
def get_train_dev_test(data_x, data_y, seed, save_test_path=None):

    # 划分训练验证与测试
    train_data_x, dev_test_data_x, train_data_y, dev_test_data_y = train_test_split(data_x, data_y, test_size=0.3, shuffle=True, stratify=data_y, random_state=seed)
    dev_data_x, test_data_x, dev_data_y, test_data_y = train_test_split(dev_test_data_x, dev_test_data_y, test_size=0.4, shuffle=True, stratify=dev_test_data_y, random_state=seed)

    # 是否保存
    if save_test_path != None:
        pickle.dump(test_data_x, open(save_test_path, 'wb'))
        pickle.dump(test_data_y, open(save_test_path, 'wb'))

    train_data_x = torch.from_numpy(train_data_x)
    train_data_y = torch.from_numpy(train_data_y)
    dev_data_x = torch.from_numpy(dev_data_x)
    dev_data_y = torch.from_numpy(dev_data_y)
    test_data_x = torch.from_numpy(test_data_x)
    test_data_y = torch.from_numpy(test_data_y)

    return train_data_x, train_data_y, dev_data_x, dev_data_y, test_data_x, test_data_y


def class_s(path):
    data_y = pickle.load(open(path, 'rb')).astype(np.float32)
    class_num = [0 for i in range(10)]
    for d in data_y:
        class_num[d.astype(np.int)] += 1
    print(class_num)


# 处理数据
def data_pcs():
    parent_dir = './data/UrbanSound8K/audio/'
    fold_prename = 'all'
    file_name = '*.wav'
    wav2np(parent_dir, fold_prename, file_name, './data/preprocess/STFT_std')

def data_pcs_one_dim():
    parent_dir = './data/UrbanSound8K/audio/'
    fold_prename = 'all'
    file_name = '*.wav'
    wav2np_one_dim(parent_dir, fold_prename, file_name, './data/preprocess/one_dim/od')

def mini_data_gen(single_data):
    k = 84
    n = 3
    mini_size = 4 * n + 6
    new_d = np.empty((0, 2, mini_size, 173))
    for i in range(k + 1):
        start = i * 12
        if(i == k):
            mini_data = single_data[:, start:, :]
            ap = np.zeros((2, 18 - mini_data.shape[1], 173))
            mini_data = np.concatenate((mini_data, ap), axis=1)
        else:
            end = start + mini_size
            mini_data = single_data[:, start:end, :]

        new_d = np.append(new_d, mini_data.reshape(1, 2, mini_size, 173), axis=0)

    return new_d

def write_test_single(data, path):
    f = open(path, 'w')
    for a in data:
        for b in a:
            for c in b:
                for d in c:
                    f.writelines(str(d) + '\n')


def generate_for_h():
    data_x, data_y = stft_general_data()
    print(data_x.shape)
    new_data = mini_data_gen(data_x[0])
    print(new_data.shape)
    write_test_single(new_data, '../hardware/resources/testdata.txt')

# 
def cpt_h_r(path, channel, height, width):
    f = open(path, 'r')
    lines = f.readlines()

    # 每个ConvModule输出长度
    per_len = channel * height * width

    data_b = []
    # 每个数据形成一个list
    for l in lines:
        data_b.append(int(l.split('(')[1].split(')')[0]))
    # 根据per_len 的长度划分小区域
    data_k = [ data_b[i:i + per_len] for i in range(int(len(data_b) / per_len))]
    fi_data = []

    # 根据三个指标将小区域格式化成c， h， w
    for d in data_k:
        new_d = np.array(d).reshape(width, channel * height)
        new_d = np.transpose(new_d)
        p = np.empty((0, height, width))
        for k in range(channel):
            p = np.append(p, new_d[k: k + height, :].reshape(1, height, width), axis=0)
        fi_data.append(p)
    result = np.array(fi_data)

    return result


if __name__ == '__main__':
    # d =  cpt_h_r('./logtest.txt', 2, 2, 7)
    # print(d.shape)
    # print(d)
    data_pcs_one_dim()