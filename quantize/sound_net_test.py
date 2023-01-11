from experiment import setup_seed, one_dim_data
from train import train
from utils.dataTrans import get_dataloader
from sklearn.model_selection import train_test_split
from model import SoundNet, SoundNet_no_padding
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

# 参数
BATCH_SIZE = 16
LR = 1e-4
EPOCH = 120
# 随机种子
seed = 7266

def train_test_divide(data_x, data_y, k, index):
    data_num = data_x.shape[0]
    one_fold_num = int(data_num / k)
    if(index >= k):
        print("index should < k")
        exit()
    if(index == k-1):
        train_data_x = data_x[0:index * one_fold_num, :, :, :]
        train_data_y = data_y[0:index * one_fold_num]
        val_data_x = data_x[index*one_fold_num:, :, :, :]
        val_data_y = data_y[index*one_fold_num:]
    else:
        train_data_x = np.append(data_x[0:index * one_fold_num, :, :, :], data_x[(index+1) * one_fold_num:, :, :, :], axis=0)
        train_data_y = np.append(data_y[0:index * one_fold_num], data_y[(index+1) * one_fold_num:], axis=0)
        val_data_x = data_x[index*one_fold_num:(index+1)*one_fold_num, :, :, :]
        val_data_y = data_y[index*one_fold_num:(index+1)*one_fold_num]
    return train_data_x, val_data_x, train_data_y, val_data_y


def k_fold(data_x, data_y, K):
    setup_seed(seed)
    # 保存相关
    save_model_path = 'models/sound'
    save_fig_path = 'fig/sound'
    model_name = 'soundnet_new_k'
    skf = StratifiedKFold(n_splits=K)
    print(data_x.shape)
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_x, data_y)):
        train_x, val_x = data_x[train_idx, ...], data_x[val_idx, ...]
        train_y, val_y = data_y[train_idx, ...], data_y[val_idx, ...]
        train_loader = get_dataloader(train_x, train_y, BATCH_SIZE, 0)
        model = SoundNet()
        # 训练
        scores_list= train(train_loader, val_x, val_y, model, LR, EPOCH, save_model_path, save_fig_path,  model_name, str(fold) + "_" + str(K), seed)
        max_acc = max(scores_list[0])
        print('Accurracy:', max_acc)

        

def sound_net_train_test():
    setup_seed(seed)
    # 保存相关
    save_model_path = 'models/sound'
    save_fig_path = 'fig/sound'
    model_name = 'soundnet_l2_new'

    k = 7
    data_x, data_y = one_dim_data()
    for index in range(k):
        # 数据
        train_data_x, dev_data_x, train_data_y, dev_data_y = train_test_divide(data_x, data_y, k, index)
        train_loader = get_dataloader(train_data_x, train_data_y, BATCH_SIZE, 0)
        model = SoundNet()
        # 训练
        scores_list= train(train_loader, dev_data_x, dev_data_y, model, LR, EPOCH, save_model_path, save_fig_path,  model_name, str(index) + "_" + str(k), seed)
        max_acc = max(scores_list[0])
        print('Accurracy:', max_acc)

if __name__ == '__main__':
    # sound_net_train_test()
    # in_data = np.random.rand(1, 1, 220050, 1)
    # print(in_data.shape)
    # model = SoundNet()
    # out = model(torch.from_numpy(in_data).float())
    data_x, data_y = one_dim_data()
    k_fold(data_x, data_y, 7)
    