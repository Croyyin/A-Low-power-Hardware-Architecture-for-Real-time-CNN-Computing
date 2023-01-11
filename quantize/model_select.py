import copy
from operator import mod 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from train import train
from model import Unquantified_2layer_CNN, VGGMini_CNN
from utils.function import direct_quantize
from utils.dataTrans import get_dataloader
from sklearn.model_selection import train_test_split
from experiment import setup_seed, stft_general_data, stft_general_test

BATCH_SIZE = 64
LR = 0.003
EPOCH = 50

# 参数

kernel_size = [3, 3, 3, 3]
stride = [1, 1, 1, 1]

pooling_stride_o = (2, 1)
pooling_stride_n = (2, 2)

feature_map_size_stft = (1025, 173)
feature_map_size_mfcc = (40, 173)

# test train
def real_train(train_data_x, dev_data_x, train_data_y, dev_data_y, seed_, model, save_model_path, save_fig_path, model_name, tag):
    # 确定随机数种子
    setup_seed(seed_)
    train_loader = get_dataloader(train_data_x, train_data_y, BATCH_SIZE)
    # 训练
    scores_list= train(train_loader, dev_data_x, dev_data_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, tag, seed_)
    max_acc = max(scores_list[0])

    return max_acc


def test_main(mode):
    seed = 7266

    # 保存相关
    save_model_path = 'models/test/stft'
    save_fig_path = 'fig/test/stft'

    # VGG_Mini
    if mode == 1:
        # 数据准备
        data_x, data_y = stft_general_data()
        train_data_x, dev_data_x, train_data_y, dev_data_y = train_test_split(data_x, data_y, test_size=0.25, shuffle=True, stratify=data_y, random_state=seed)
        # 模型确定
        model = VGGMini_CNN([2, 16, 16, 32], [16, 16, 32, 32], kernel_size, stride, pooling_stride_n, feature_map_size_stft)
        model_name_o = 'VGGMini'

        print('对于STFT数据集')
    else:
        train_data_x, train_data_y = stft_general_data()
        dev_data_x, dev_data_y = stft_general_test()


    
    max_acc = real_train(train_data_x, dev_data_x, train_data_y, dev_data_y, seed, model, save_model_path, save_fig_path, model_name_o, '32_64')
    print('Accurracy:', max_acc)



if __name__ == '__main__':
    test_main(1)


