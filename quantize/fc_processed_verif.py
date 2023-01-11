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
from spatial_test import load_data

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

# 数据加载
def data_clp():
    data_x, data_y = load_data('data/preprocess/long_STFT_x.dat', 'data/preprocess/long_STFT_y.dat')
    new_data_x = []
    for i in range(173):
        new_data_x.append(data_x[:, :, :, i:i+173])
        
    return new_data_x, data_y

# 计算并保存运行结果
def inference_probability_normal(data_x_list, data_y, save_path):
    # shape of input x: (time, data_num, )
    # 模型准备
    model_n = Unquantified_2layer_CNN(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_n, feature_map_size)
    model_n.load_state_dict(torch.load('models/very/stft/noverlap/1/best_model7266.pt'))

    model_o = Unquantified_2layer_CNN(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_o, feature_map_size)
    model_o.load_state_dict(torch.load('models/very/stft/overlap/1/best_model7266.pt'))

    # 统计相关参数
    entier_time_category_o = []
    entier_time_category_n = []

    for data_x in tqdm(data_x_list, desc='time'):
        data_x = torch.from_numpy(data_x)
        # overlapping pooling 预测
        predict_o = model_o(data_x)
        # 归入总数据
        entier_time_category_o.append(predict_o.detach().numpy())
        # 与上述相同
        predict_n = model_n(data_x)
        # 归入总数据
        entier_time_category_n.append(predict_n.detach().numpy())

    # 运行结果保存
    pickle.dump(entier_time_category_o, open(save_path + 'ten_classes_probability_normal_overlap.dat', 'wb'))
    pickle.dump(entier_time_category_n, open(save_path + 'ten_classes_probability_normal_noverlap.dat', 'wb'))

    return entier_time_category_o, entier_time_category_n
    
# 计算并保存运行结果
def inference_probability_fc_processed(data_x_list, data_y, save_path):
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
    
    model_o.eval()
    model_n.eval()
    with torch.no_grad():
        for data_x in tqdm(data_x_list, desc='time'):
            data_x = torch.from_numpy(data_x)
            # overlapping pooling 预测
            predict_o, result_o, column_o = model_o.fc_forward(data_x, result_o, column_o, mode)
            # 归入总数据
            entier_time_category_o.append(predict_o.detach().numpy())

            # 与上述相同
            predict_n, result_n, column_n = model_n.fc_forward(data_x, result_n, column_n, mode)
            # 归入总数据
            entier_time_category_n.append(predict_n.detach().numpy())
            result_o.detach()
            column_o.detach()

            if mode == 1:
                mode = 0
    
    # 运行结果保存
    pickle.dump(entier_time_category_o, open(save_path + 'ten_classes_probability_fc_processed_overlap.dat', 'wb'))
    pickle.dump(entier_time_category_n, open(save_path + 'ten_classes_probability_fc_processed_noverlap.dat', 'wb'))

    return entier_time_category_o, entier_time_category_n
    
def fig_print_all(data_y, trlo, trln, save_name):

    trlo = np.array(trlo).astype(np.float32)
    trln = np.array(trln).astype(np.float32)
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

            plt.savefig('fig/'+ save_name + str(i) + '.svg', bbox_inches='tight')
            ax[1].cla()
            ax[0].cla()


def count_rate(data_y, trlo_normal, trln_normal, trlo_fc, trln_fc):
    # normal
    trlo_normal = np.array(trlo_normal).astype(np.float32)
    trln_normal = np.array(trln_normal).astype(np.float32)
    trlo_normal = [ trlo_normal[:, i, :].reshape(trlo_normal.shape[0], -1).T for i in range(trlo_normal.shape[1])]
    trln_normal = [ trln_normal[:, i, :].reshape(trln_normal.shape[0], -1).T for i in range(trln_normal.shape[1])]
    # fc
    trlo_fc = np.array(trlo_fc).astype(np.float32)
    trln_fc = np.array(trln_fc).astype(np.float32)
    trlo_fc = [ trlo_fc[:, i, :].reshape(trlo_fc.shape[0], -1).T for i in range(trlo_fc.shape[1])]
    trln_fc = [ trln_fc[:, i, :].reshape(trln_fc.shape[0], -1).T for i in range(trln_fc.shape[1])]

    # count
    right_count_normal = 0
    right_count_fc = 0
    same_count = 0
    #
    right_count_o_normal = 0
    right_count_n_normal = 0
    right_count_o_fc = 0
    right_count_n_fc = 0
    #
    same_count_o = 0
    same_count_n = 0


    total = len(trlo_normal)

    for i in tqdm(range(total)):
        # normal
        final_o_list_normal = trlo_normal[i][:, -1].reshape(-1).tolist()
        max_o_normal = final_o_list_normal.index(max(final_o_list_normal))
        
        final_n_list_normal = trln_normal[i][:, -1].reshape(-1).tolist()
        max_n_normal = final_n_list_normal.index(max(final_n_list_normal))

        # fc processed
        final_o_list_fc = trlo_fc[i][:, -1].reshape(-1).tolist()
        max_o_fc = final_o_list_fc.index(max(final_o_list_fc))
        
        final_n_list_fc = trln_fc[i][:, -1].reshape(-1).tolist()
        max_n_fc = final_n_list_fc.index(max(final_n_list_fc))

        # 正常情况下, 重叠与非重叠的同时正确
        if max_o_normal == max_n_normal and max_n_normal == data_y[i]:
            right_count_normal += 1
        
        # 全连接层处理情况下, 重叠与非重叠的同时正确
        if max_o_fc == max_n_fc and max_n_fc == data_y[i]:
            right_count_fc += 1
        
        # 正常与全连接层处理相同
        if max_o_fc == max_n_fc and max_o_normal == max_n_normal and max_n_fc == data_y[i]:
            same_count += 1

        # 正常情况下, 仅重叠正确
        if max_o_normal == data_y[i]:
            right_count_o_normal += 1
        # 正常情况下, 仅非重叠正确
        if max_n_normal == data_y[i]:
            right_count_n_normal += 1
        # 全连接层处理情况下, 仅重叠正确
        if max_o_fc  == data_y[i]:
            right_count_o_fc += 1
        # 全连接层处理情况下, 仅非重叠正确
        if max_n_fc == data_y[i]:
            right_count_n_fc += 1

        # 正常与全连接层处理相同, 重叠情况下
        if max_o_fc == max_o_normal and max_o_fc == data_y[i]:
            same_count_o += 1
        # 正常与全连接层处理相同, 非重叠情况下
        if max_n_fc == max_n_normal and max_n_fc == data_y[i]:
            same_count_n += 1

    coacc_normal_o = right_count_o_normal / total
    coacc_normal_n = right_count_n_normal / total
    coacc_normal = right_count_normal / total

    coacc_fc_o = right_count_o_fc / total
    coacc_fc_n = right_count_n_fc / total
    coacc_fc = right_count_fc / total
    print('正常情况下, 仅重叠正确的比率:', coacc_normal_o)
    print('正常情况下, 仅非重叠正确的比率:', coacc_normal_n)
    print('正常情况下, 重叠与非重叠的同时正确的比率:', coacc_normal)
    print('全连接层处理情况下, 仅重叠正确的比率:', coacc_fc_o)
    print('全连接层处理情况下, 仅非重叠正确的比率:', coacc_fc_n)
    print('全连接层处理情况下, 重叠与非重叠的同时正确的比率:', coacc_fc)


    same_in_o_normal = same_count_o / right_count_o_normal
    same_in_n_normal = same_count_n / right_count_n_normal
    same_in_normal = same_count / right_count_normal

    same_in_o_fc = same_count_o / right_count_o_fc
    same_in_n_fc = same_count_n / right_count_n_fc
    same_in_fc = same_count / right_count_fc

    print('仅重叠, 正常与全连接层处理相同占正常情况的比率:', same_in_o_normal)
    print('仅非重叠, 正常与全连接层处理相同占正常情况的比率:', same_in_n_normal)
    print('正常与全连接层处理相同占正常情况的比率:', same_in_normal)
    print('仅重叠, 正常与全连接层处理相同占全连接层处理情况的比率', same_in_o_fc)
    print('仅非重叠, 正常与全连接层处理相同占全连接层处理情况的比率', same_in_n_fc)
    print('正常与全连接层处理相同占全连接层处理情况的比率', same_in_fc)
    print(total)
    

    

    


def main():
    # 数据准备
    data_x_list, data_y = data_clp()
    data_x_list = np.array(data_x_list)
    data_y = np.array(data_y)
    print('start process')
    # result_o, resutl_n = inference_probability_normal(data_x_list, data_y, 'data/')
    # fig_print_all(data_y, result_o, resutl_n, 'normal/all_true_n')
    # result_o_fc, resutl_n_fc = inference_probability_fc_processed(data_x_list, data_y, 'data/')
    # fig_print_all(data_y, result_o_fc, resutl_n_fc, 'fc_processed/all_true_fc')

    trlo_normal = pickle.load(open('data/ten_classes_probability_normal_overlap.dat', 'rb'))
    trln_normal = pickle.load(open('data/ten_classes_probability_normal_noverlap.dat', 'rb'))
    trlo_fc = pickle.load(open('data/ten_classes_probability_fc_processed_overlap.dat', 'rb'))
    trln_fc = pickle.load(open('data/ten_classes_probability_fc_processed_noverlap.dat', 'rb'))
    # fig_print_all(data_y, trlo_fc, trln_fc, 'fc_processed/all_true_fc')
    # fig_print_all(data_y, trlo_normal, trln_normal, 'normal/all_true_n')
    count_rate(data_y, trlo_normal, trln_normal, trlo_fc, trln_fc)

if __name__ == '__main__':
    main()
    
    