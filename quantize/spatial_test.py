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

# audio
import librosa
import librosa.display
import sklearn
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from model import Unquantified_2layer_CNN, Unquantified_2layer_CNN_sp

# 参数
in_channels_list = [2, 1]
out_channels_list =[1, 1]
kernel_size = [3, 3]
stride = [1, 1]
pooling_stride_o = (2, 1)
pooling_stride_n = (2, 2)
feature_map_size = (1025, 173)
BATCH_SIZE = 64


# 读取采样, 返回采样频率与原始数据
def load_clip(filename):
    x, sr = librosa.load(filename, sr=22050)
    if 4 * sr - x.shape[0] > 0: 
        x = np.pad(x,(0,4 * sr - x.shape[0]),'constant')
    else:
        x = x[:4 * sr]

    x = x.tolist()
    return x, sr

# STFT特征提取，双通道
def extract_feature_stft(x):
    stft = librosa.stft(x)
    stft_abs = np.abs(stft).astype(float).reshape(1, 1025, 345)
    stft_angle = np.angle(stft).astype(float).reshape(1, 1025, 345)

    return stft_abs, stft_angle

# 数据读取，返回所有原始数据的data和label
def load_all_origin(filenames):
    data_x, data_y = [], []
    for filename in tqdm(filenames, desc='sample'):

        x, _ = load_clip(filename)
        data_x.append(x)
        data_y.append(int(filename.split('/')[-1].split('-')[1]))

    return data_x, data_y

# 每个类别选出100个数据，十分类
def h_select(data_x, data_y, n=100):
    data_y, data_x = zip(*sorted(zip(data_y, data_x)))
    new_data_x, new_data_y = [], []
    for i in range(10):
        index = data_y.index(i)
        new_data_x.append(data_x[index: index + n])
        new_data_y.append([ i for j in range(n)])

    return new_data_x, new_data_y

# 拼接两类原始数据，并以后一类数据的label作为真实label
def ctnt_data(data_x, data_y):
    random_list = [ random.sample(range(0, 3), 2) for i in range(10)]
    new_data_x, new_data_y = [], []
    for i in tqdm(range(len(random_list)), desc='sample ctnt'):
        apt = []
        for j in range(len(data_x[0])):
            apt.append(data_x[random_list[i][0]][j] + data_x[random_list[i][1]][j])

        new_data_x.append(apt)
        new_data_y.append(data_y[random_list[i][1]])

    return new_data_x, new_data_y

# 产生拼接数据集，返回处理后结果
def load_dataset_stft(filenames):
    # 预处理
    data_x, data_y = load_all_origin(filenames)
    data_x, data_y = h_select(data_x, data_y)
    data_x, data_y = ctnt_data(data_x, data_y)

    real_d_x, real_d_y = [], []
    # 数据重置为一列
    for d1 in data_x:
        for d2 in d1:
            real_d_x.append(d2)
    for di in data_y:
        for dd in di:
            real_d_y.append(dd)

    real_d_x = np.array(real_d_x)
    # 特征提取
    features = np.empty((0, 2, 1025, 345))
    for f in tqdm(real_d_x, desc='stft'):
        one_sample = np.empty((0, 1025, 345))
        stft_real, stft_abv = extract_feature_stft(f)
        one_sample = np.append(one_sample, stft_real, axis=0)
        one_sample = np.append(one_sample, stft_abv, axis=0)
        
        features = np.append(features, one_sample.reshape(1, 2, 1025, 345), axis=0)

    return np.array(features), np.array(real_d_y, dtype=np.int)

# 顶层函数，将wav文件转化为numpy
def wav2np(parent_dir, fold_prename, file_name, save_path_prename):
    # 加载所有数据文件
    data_files = glob.glob(os.path.join(parent_dir, fold_prename + '*', file_name))

    # 数据转换
    data_x, data_y = load_dataset_stft(data_files)
    pickle.dump(data_x, open(save_path_prename + '_x.dat', 'wb'))
    pickle.dump(data_y, open(save_path_prename + '_y.dat', 'wb'))

# 顶层函数，调用wav2np
def data_pcs():
    parent_dir = './data/UrbanSound8K/audio/'
    fold_prename = 'all'
    file_name = '*.wav'
    wav2np(parent_dir, fold_prename, file_name, './data/preprocess/long_STFT')

# 数据加载
def load_data(data_x_path, data_y_path):
    data_x = pickle.load(open(data_x_path, 'rb')).astype(np.float32)
    data_y = pickle.load(open(data_y_path, 'rb')).astype(np.float32)
    return data_x, data_y

# 加载处理后的数据，每个长度为345的数据分化为173个173
def data_clp():
    data_x, data_y = load_data('data/preprocess/long_STFT_x.dat', 'data/preprocess/long_STFT_y.dat')
    new_data_x = []
    for i in range(173):
        new_data_x.append(data_x[:, :, :, i:i+173])
        
    return new_data_x, data_y

# 计算并保存运行结果
def infrc():
    # 数据准备
    data_x_list, data_y = data_clp()
    data_x_list = np.array(data_x_list)
    data_y = np.array(data_y)

    # 模型准备
    model_n = Unquantified_2layer_CNN(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_n, feature_map_size)
    model_n.load_state_dict(torch.load('models/very/stft/noverlap/1/best_model7266.pt'))

    model_o = Unquantified_2layer_CNN(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_o, feature_map_size)
    model_o.load_state_dict(torch.load('models/very/stft/overlap/1/best_model7266.pt'))

    # 统计相关参数
    entier_time_category_o = []
    entier_time_category_n = []


    for data_x_n in tqdm(data_x_list, desc='time'):
        data_x = torch.from_numpy(data_x_n)
        # overlapping pooling 预测
        predict_o = model_o(data_x)

        # 获得对应分类的概率
        # predict_probability_o = predict_o[np.arange(data_y.shape[0]), data_y]
        
        # 归入总数据
        entier_time_category_o.append(predict_o.detach().numpy())

        # 与上述相同
        predict_n = model_n(data_x)
        # predict_probability_n = predict_n[np.arange(data_y.shape[0]), data_y]

        # 归入总数据
        entier_time_category_n.append(predict_n.detach().numpy())

    
    # 运行结果保存
    pickle.dump(entier_time_category_o, open('data/ten_classes_all_overlap.dat', 'wb'))
    pickle.dump(entier_time_category_n, open('data/ten_classes_all_noverlap.dat', 'wb'))
    
# 计算并保存运行结果
def infrc_fc_sp():
    # 数据准备
    data_x_list, data_y = data_clp()
    data_x_list = np.array(data_x_list)[:,0:10,:,:]
    data_y = np.array(data_y)[0:10]

    # 模型准备
    model_n = Unquantified_2layer_CNN_sp(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_n, feature_map_size)
    model_n.load_state_dict(torch.load('models/very/stft/noverlap/1/best_model7266.pt'))

    model_o = Unquantified_2layer_CNN_sp(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_o, feature_map_size)
    model_o.load_state_dict(torch.load('models/very/stft/overlap/1/best_model7266.pt'))

    # 统计相关参数
    entier_time_category_o = []
    entier_time_category_n = []
    mode = 1
    column_o = None
    column_n = None
    result_o = None
    result_n = None

    for data_x_n in tqdm(data_x_list, desc='time'):
        data_x = torch.from_numpy(data_x_n)
        # overlapping pooling 预测
        predict_o, result_o, column_o = model_o.fc_forward(data_x, result_o, column_o, mode)
        # 获得对应分类的概率
        # predict_probability_o = predict_o[np.arange(data_y.shape[0]), data_y]
        
        # 归入总数据
        entier_time_category_o.append(predict_o.detach().numpy())

        # 与上述相同
        predict_n, result_n, column_n = model_n.fc_forward(data_x, result_n, column_n, mode)
        # predict_probability_n = predict_n[np.arange(data_y.shape[0]), data_y]

        # 归入总数据
        entier_time_category_n.append(predict_n.detach().numpy())
        if mode == 1:
            mode = 0
    
    # 运行结果保存
    pickle.dump(entier_time_category_o, open('data/ten_classes_sp_overlap.dat', 'wb'))
    pickle.dump(entier_time_category_n, open('data/ten_classes_sp_noverlap.dat', 'wb'))
    

def fig_print():
    
    trlo = pickle.load(open('data/mini_sp_overlap.dat', 'rb')).astype(np.float32).tolist()
    trln = pickle.load(open('data/mini_sp_noverlap.dat', 'rb')).astype(np.float32).tolist()

    # 图
    color_list = ['cornflowerblue', 'tomato', 'gold', 'springgreen']
    marker_list = ['*', 'v', '2', 's', 'X', 'P', 'd', 'p', '|', 'o']
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    plt.rcParams.update({"font.size": 13})

    time = [ i for i in range(173)]

    for i in tqdm(range(len(trlo))):
        # 
        ax[1].set_title('Overlap')
        ax[1].plot(time, trlo[i], c=color_list[0], label = 'overlap')
        ax[1].legend()

        ax[0].set_title('Non Overlap')
        ax[0].plot(time, trln[i], c=color_list[1], label = 'noverlap')
        ax[0].legend()

        plt.savefig('fig/pt_sp' + str(i) + '.svg', bbox_inches='tight')
        ax[1].cla()
        ax[0].cla()


def fig_print_all():
    _, data_y = data_clp()
    data_y = np.array(data_y)[0:10]

    trlo = np.array(pickle.load(open('data/ten_classes_sp_overlap.dat', 'rb'))).astype(np.float32)
    trln = np.array(pickle.load(open('data/ten_classes_sp_noverlap.dat', 'rb'))).astype(np.float32)
    trlo = [ trlo[:, i, :].reshape(trlo.shape[0], -1).T for i in range(trlo.shape[1])]
    trln = [ trln[:, i, :].reshape(trln.shape[0], -1).T for i in range(trln.shape[1])]

    # 图
    color_list = ['darkgrey', 'tomato', 'darkorange', 'gold', 'limegreen', 'darkturquoise', 'cornflowerblue', 'darkorchid', 'deeppink', 'indigo']
    marker_list = ['*', 'v', '2', 's', 'X', 'P', 'd', 'p', '|', 'o']
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    plt.rcParams.update({"font.size": 13})

    time = [ i for i in range(173)]

    for i in tqdm(range(len(trlo))):
        # 
        final_o_list = trlo[i][:, -1].reshape(-1).tolist()
        max_o = final_o_list.index(max(final_o_list))
        
        final_n_list = trln[i][:, -1].reshape(-1).tolist()
        max_n = final_n_list.index(max(final_n_list))
        if max_o == max_n and max_n == data_y[i]:

            ax[1].set_title('Overlap')
            for j in range(trlo[i].shape[0]):
                adt = np.ones(trlo[i].shape[1])
                ax[1].plot(time, trlo[i][j] + adt * j, c=color_list[j], label = 'Class '+str(j  +1))

            ax[1].legend(bbox_to_anchor=(1.1, 0), loc=3, borderaxespad=0)

            ax[0].set_title('Non Overlap')
            for j in range(trln[i].shape[0]):
                adt = np.ones(trlo[i].shape[1])
                ax[0].plot(time, trln[i][j] + adt * j, c=color_list[j], label = 'Class '+str(j  +1))
            # ax[0].legend()

            plt.savefig('fig/all_classes_ture_sp' + str(i) + '.svg', bbox_inches='tight')
            ax[1].cla()
            ax[0].cla()


def updown():
    trlo = pickle.load(open('data/1000_overlap.dat', 'rb')).astype(np.float32).tolist()
    trln = pickle.load(open('data/1000_noverlap.dat', 'rb')).astype(np.float32).tolist()

    upd_num = 0
    ovflg = 0
    n_0 = 0
    for i in range(len(trlo)):
        # 起伏次数
        o_flag = 0
        n_flag = 0
        # 状态初始化
        o_last = trlo[i][0]
        n_last = trln[i][0]


        for j in range(len(trlo[i])):
            # overlap
            if o_last == trlo[i][j]:
                # 上个状态与当前状态相同：未波动
                pass
            else:
                # 上一个状态与当前状态不同：单次波动
                # 将状态置为当前状态
                o_last = trlo[i][j]

                # 波动标记自增
                o_flag += 1

            # non overlap
            if n_last == trln[i][j]:
                pass
            else:
                n_last = trln[i][j]
                n_flag += 1

        # 判断当前数据非重叠池化的波动次数是否超过重叠池化
        if (n_flag > o_flag) :
            # 
            upd_num += 1
        if o_flag < 4:
            ovflg += 1

        if n_flag > o_flag and o_flag < 4:
            n_0 += 1

    upd_rate = upd_num / len(trlo)
    ov_rate = ovflg / len(trlo)
    n_0_rate = n_0 / ovflg
    print(upd_rate)
    print(ov_rate)
    print(n_0_rate)
    

def main():
    data_pcs()

if __name__ == '__main__':
    # infrc()
    # infrc_fc_sp()
    fig_print_all()
    # fig_print()
    
    