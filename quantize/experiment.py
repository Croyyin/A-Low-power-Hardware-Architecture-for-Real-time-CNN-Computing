import copy 
import pickle
import pulp
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from numpy.random.mtrand import f
from sklearn.metrics import accuracy_score
import torch
import math
from tqdm import tqdm
from train import train
from data_preprocess import wav2np
from math import floor, log, pi
from model import CNN_2layer_1channel_1channel_3k_1linear, CNN_2layer_1channel_1channel_3k_2linear, Quantified_2layer_CNN, Unquantified_2layer_CNN
from utils.function import direct_quantize
from utils.dataTrans import get_data, get_dataloader
from data_preprocess import get_train_dev_test, load_data
from sklearn.model_selection import train_test_split


# 设置随机
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 选用部分数据集
def get_partial_data_x(data, num):
    if num == 1025:
        return data

    final_data = data[:, :, 0, :]
    final_data = final_data.reshape(final_data.shape[0], final_data.shape[1], 1, final_data.shape[2])

    step = math.floor(1025 / num)
    for i in range(1, num):
        d = data[:, :, i * step, :]
        d = d.reshape(d.shape[0], d.shape[1], 1, d.shape[2])
        final_data = np.concatenate((final_data, d), axis=2)

    return final_data

# stft通用训练验证集
def stft_general_data():
    # 加载数据
    data_x, data_y = load_data('data/preprocess/stft/std/STFT_std_train_and_dev_v1_x.dat', 'data/preprocess/stft/std/STFT_std_train_and_dev_v1_y.dat')
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    return data_x, data_y

def one_dim_data():
    # 加载数据
    data_x, data_y = load_data('data/preprocess/one_dim/od_one_dim_x.dat', 'data/preprocess/one_dim/od_one_dim_y.dat')
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    return data_x, data_y

def stft_general_test():
    # 加载数据
    data_x, data_y = load_data('data/preprocess/stft/n/STFT_test_v1_x.dat', 'data/preprocess/stft/n/STFT_test_v1_y.dat')
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    return data_x, data_y

# mfcc通用训练验证集
def mfcc_general_data():
    # 加载数据
    data_x, data_y = load_data('data/preprocess/mfcc/std/MFCC_std_train_and_dev_v1_x.dat', 'data/preprocess/mfcc/std/MFCC_std_train_and_dev_v1_y.dat')
    data_x = data_x.reshape(data_x.shape[0], -1, data_x.shape[1], data_x.shape[2])
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    return data_x, data_y

def mfcc_general_test():
    # 加载数据
    data_x, data_y = load_data('data/preprocess/mfcc/std/MFCC_std_test_v1_x.dat', 'data/preprocess/mfcc/std/MFCC_std_test_v1_y.dat')
    data_x = data_x.reshape(data_x.shape[0], -1, data_x.shape[1], data_x.shape[2])
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    return data_x, data_y

# 测试集划分
def test_set_cfm():
    data_x, data_y = load_data('data/preprocess/stft/std/STFT_std_x.dat', 'data/preprocess/stft/std/STFT_std_y.dat')
    train_and_dev_x, test_x, train_and_dev_y, test_y = train_test_split(data_x, data_y, test_size=0.1, shuffle=True, stratify=data_y, random_state=291106128)

    pickle.dump(train_and_dev_x, open('./data/preprocess/stft/std/STFT_std_train_and_dev_v1' + '_x.dat', 'wb'))
    pickle.dump(train_and_dev_y, open('./data/preprocess/stft/std/STFT_std_train_and_dev_v1' + '_y.dat', 'wb'))
    pickle.dump(test_x, open('./data/preprocess/stft/std/STFT_std_test_v1' + '_x.dat', 'wb'))
    pickle.dump(test_y, open('./data/preprocess/stft/std/STFT_std_test_v1' + '_y.dat', 'wb'))


# 交叉验证
def cross():
    # 参数
    in_channels_list = [2, 4]
    out_channels_list =[4, 4]
    kernel_size = [3, 3]
    stride = [1, 1]
    pooling_stride_n = (2, 2)
    feature_map_size = (1025, 173)
    BATCH_SIZE = 64
    LR = 0.001
    EPOCH = 50
    k_fold = 7

    # 加载数据
    train_data_x, train_data_y = load_data('data/preprocess/STFT_train_and_dev_v1_x.dat', 'data/preprocess/STFT_train_and_dev_v1_y.dat')
    dev_test_x, dev_test_y = load_data('data/preprocess/STFT_test_v1_x.dat', 'data/preprocess/STFT_test_v1_y.dat')

    seed_list = [random.randint(10000, 100000) for i in range(4)]

    data_size = train_data_x.shape[0]
    per_fold_num = floor(data_size / k_fold)
    data_x_list = []
    data_y_list = []

    for i in range(k_fold):
        if i == k_fold - 1:
            data_x_list.append(train_data_x[i * per_fold_num:])
            data_y_list.append(train_data_y[i * per_fold_num:])
        else:
            data_x_list.append(train_data_x[i * per_fold_num:(i + 1) * per_fold_num])
            data_y_list.append(train_data_y[i * per_fold_num:(i + 1) * per_fold_num])

    for i in range(k_fold):
        dev_data_x = data_x_list[i]
        dev_data_y = data_y_list[i]
        dev_data_x = torch.from_numpy(dev_data_x)
        dev_data_y = torch.from_numpy(dev_data_y)


        train_data_x = np.empty((0, 2, 1025, 173))
        train_data_y = np.empty(0)

        for j in range(k_fold):
            if j != i:
                train_data_x = np.append(train_data_x, data_x_list[j], axis=0)
                train_data_y = np.append(train_data_y, data_y_list[j], axis=0)
        
        train_data_x = torch.from_numpy(train_data_x.astype(np.float32))
        train_data_y = torch.from_numpy(train_data_y.astype(np.float32))

        for k in range(len(seed_list)):
            # 确定随机数种子
            setup_seed(seed_list[k])
            train_loader = get_dataloader(train_data_x, train_data_y, BATCH_SIZE)
            # 不采用overlap
            model_n = Unquantified_2layer_CNN(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_n,     feature_map_size)
            # 训练
            train(train_loader, dev_data_x, dev_data_y, model_n, LR, EPOCH, 'models', 'fig', 'cross', i + 1, seed_list[k])

# STFT和CNN计算量
def stft_cnn_mnum():
    max_window = 20
    inchl = 3
    outchl = 5
    alpha = inchl * outchl
    sample = 88200
    color_list = ['red', 'darkorange', 'olivedrab', 'dimgray', 'deeppink']

    # second chart
    # STFT
    STFT_x_1 = [i + 1 for i in range(max_window)]
    STFT_y_1 = [log(2) for i in range(max_window)]
    STFT_y_2 = [log(13) for i in range(max_window)]

    # CONV
    CONV_ws_x = np.arange(1, max_window + 1, 1).astype(np.float32)
    CONV_ws_y_1 = np.log(CONV_ws_x / 2)
    CONV_ws_y_2 = np.log(CONV_ws_x / 2 * 15) 

    # plot
    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({"font.size": 13})
    ax = fig.gca()
    
    plt.xlabel('kernel^2 / stride')
    plt.ylabel('ln(number of multiplications)')


    ax.plot(STFT_x_1, STFT_y_1, c='cornflowerblue', label = 'STFT')
    ax.plot(STFT_x_1, STFT_y_2, c='cornflowerblue')
    ax.fill_between(STFT_x_1, STFT_y_1, STFT_y_2, color='cornflowerblue', alpha='0.5')

    ax.plot(CONV_ws_x, CONV_ws_y_1, c='darkorange', label = 'CNN')
    ax.plot(CONV_ws_x, CONV_ws_y_2, c='darkorange')
    ax.fill_between(CONV_ws_x, CONV_ws_y_1, CONV_ws_y_2, color='darkorange', alpha='0.5')
    
    plt.legend()
    plt.savefig('fig/MultyNum.svg')
    plt.show()

#
def fmp_split():
    ProbLP1 = pulp.LpProblem("ProbLP1", sense=pulp.LpMaximize)

    k = pulp.LpVariable('k', lowBound=0, upBound=1000, cat='Integer') 
    n = pulp.LpVariable('n', lowBound=2, upBound=100, cat='Integer')

    ProbLP1 += (6 + k * 4 * n)
    ProbLP1 += (6 + k * 4 * n <= 1025)

    ProbLP1.solve()
    print(ProbLP1.name)  # 输出求解状态
    print("Status:", pulp.LpStatus[ProbLP1.status])  # 输出求解状态
    for v in ProbLP1.variables():
        print(v.name, "=", v.varValue)  # 输出每个变量的最优值
    print("F1(x) =", pulp.value(ProbLP1.objective))


# 
def pic_o():
    # X
    X = np.arange(1, 80, 1).astype(np.float32)

    # Y
    S_y = X * 6.8 + 4
    H_y = X * 1.3 + 200
    K = X - 40
    J_y = np.log(K) + 100



    # plot
    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({"font.size": 13})
    ax = fig.gca()
    
    plt.xlabel('Time')
    plt.ylabel('Calculate ability')


    ax.plot(X, S_y, c='red', label = 'Software')
    ax.plot(X, H_y, c='blue', label = 'Hardware')
    ax.plot(X, J_y, c='green', label = 'Test')
    
    ax.set_xticks([])
    ax.set_yticks([])

    plt.legend()
    plt.savefig('fig/HardSoft.svg')
    plt.show()

if __name__ == '__main__':
    fmp_split()