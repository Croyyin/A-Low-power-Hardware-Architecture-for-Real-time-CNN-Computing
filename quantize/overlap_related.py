from numpy import random
from train import train
from tqdm import tqdm
from model import Unquantified_2layer_CNN
from utils.dataTrans import get_dataloader
from sklearn.model_selection import train_test_split
from experiment import setup_seed, mfcc_general_data, stft_general_data

BATCH_SIZE = 64
LR = 0.003
EPOCH = 50

# 参数
in_channels_list_stft = [[2, 1], [2, 2], [2, 4], [2, 8], [2, 16], [2, 32]]
in_channels_list_mfcc = [[1, 1], [1, 2], [1, 4], [1, 8], [1, 16], [1, 32]]
out_channels_list =[[1, 1], [2, 2], [4, 4], [8, 8], [16, 16], [32, 32]]

kernel_size = [3, 3]
stride = [1, 1]

pooling_stride_o = (2, 1)
pooling_stride_n = (2, 2)

feature_map_size_stft = (1025, 173)
feature_map_size_mfcc = (40, 173)

# overlap for stft
def train_and_score(data_x, data_y, seed_list, in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size, save_model_path, save_fig_path, model_name, tag):
    # 最大正确率
    max_acc = []

    # 种子列表循环
    for i, seed_ in tqdm(enumerate(seed_list), total=len(seed_list), desc='Total'):
        # 依据随机数划分数据
        train_data_x, dev_data_x, train_data_y, dev_data_y = train_test_split(data_x, data_y, test_size=0.25, shuffle=True, stratify=data_y, random_state=seed_)
        # 确定随机数种子
        setup_seed(seed_)
        train_loader = get_dataloader(train_data_x, train_data_y, BATCH_SIZE)
        # 
        model = Unquantified_2layer_CNN(in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size)
        # 训练
        scores_list= train(train_loader, dev_data_x, dev_data_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, tag, seed_)
        max_acc.append(max(scores_list[0]))

    return max_acc


# main
def overlap_acc_test(data_name, i):
    seed_list = [random.randint(100, 10000) for i in range(70)]
    # 保存相关
    save_model_path = ''
    save_fig_path = ''

    model_name_n = 'none_overlap'
    model_name_o = 'overlap'
    tag = out_channels_list[i][0]
    feature_map_size = []
    data_x, data_y = [], []

    if data_name == 'stft':
        # 数据
        data_x, data_y = stft_general_data()
        save_model_path = 'models/test/stft'
        save_fig_path = 'fig/test/stft'
        feature_map_size = feature_map_size_stft
        in_channels = in_channels_list_stft[i]
        print('对于STFT数据集')
    elif data_name == 'mfcc':
        # 数据
        data_x, data_y = mfcc_general_data()
        save_model_path = 'models/test/mfcc'
        save_fig_path = 'fig/test/mfcc'
        feature_map_size = feature_map_size_mfcc
        in_channels = in_channels_list_mfcc[i]
        print('对于MFCC数据集')

    # 训练求效果
    none_acc_list = train_and_score(data_x, data_y, seed_list, in_channels, out_channels_list[i], kernel_size, stride, pooling_stride_n, feature_map_size, save_model_path, save_fig_path, model_name_n, tag)
    overlap_acc_list = train_and_score(data_x, data_y, seed_list, in_channels, out_channels_list[i], kernel_size, stride, pooling_stride_o, feature_map_size, save_model_path, save_fig_path, model_name_o, tag)
    
    # 求最值
    none_max_acc = max(none_acc_list)
    overlap_max_acc = max(overlap_acc_list)
    index_n = none_acc_list.index(none_max_acc)
    index_o = overlap_acc_list.index(overlap_max_acc)
    # 求比例
    rate = (none_max_acc - overlap_max_acc) / none_max_acc
    print('非重叠池化最好效果:', none_max_acc, '种子:', seed_list[index_n])
    print('重叠池化最好效果:', overlap_max_acc, '种子:', seed_list[index_o])
    print('采用重叠池化后效果下降百分比:', rate)

if __name__ == '__main__':
    overlap_acc_test('stft', 1)