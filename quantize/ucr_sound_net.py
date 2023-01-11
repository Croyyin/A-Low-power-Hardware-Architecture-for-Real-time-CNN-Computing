from experiment import setup_seed, one_dim_data
from train import train
from utils.dataTrans import get_dataloader
from sklearn.model_selection import train_test_split
from model import SoundNet, SoundNet_no_padding, SoundNet_UCR, SoundNet_UCR_np, SoundNet_UCR_np_small
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from ucr_data_tool import class_label_make, data_read
import os
# 参数
BATCH_SIZE = 8
LR = 1e-6
EPOCH = 400
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


def k_fold(data_x, data_y, K, l2c):
    setup_seed(seed)
    # 保存相关
    save_model_path = 'models/sound'
    save_fig_path = 'fig/sound'
    model_name = 'soundnet_test'
    skf = StratifiedKFold(n_splits=K)
    h = data_x.shape[2]
    channel = data_x.shape[1]
    classes = len(l2c)
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_x, data_y)):
        train_x, val_x = data_x[train_idx, ...], data_x[val_idx, ...]
        train_y, val_y = data_y[train_idx, ...], data_y[val_idx, ...]
        train_loader = get_dataloader(train_x, train_y, BATCH_SIZE, 0)
        model = SoundNet_UCR(h, classes, channel)
        # 训练
        scores_list= train(train_loader, val_x, val_y, model, LR, EPOCH, save_model_path, save_fig_path,  model_name, str(fold) + "_" + str(K), seed)
        max_acc = max(scores_list[0])
        print('Padding Accurracy:', max_acc)
        
def main(path_data, data_set_name):
    
    setup_seed(seed)
    # 保存相关
    save_model_path = 'models/sound'
    save_fig_path = 'fig/sound'
    model_name = 'sound_net_UCR_new'
    c2l, l2c = class_label_make(path_data + "TRAIN.arff")

    data_x, data_y = data_read(path_data + "TRAIN.arff", c2l)
    data_x = data_x.astype(np.float32)
    val_x, val_y = data_read(path_data + "TEST.arff", c2l)
    val_x = val_x.astype(np.float32)
    if True in np.isnan(data_x) or True in np.isnan(val_x):
        return
    val_x, val_y = torch.from_numpy(val_x), torch.from_numpy(val_y)

    h = data_x.shape[2]
    channel = data_x.shape[1]
    classes = len(l2c)
    print("Current Dataset:", data_set_name)
    train_loader = get_dataloader(data_x, data_y, BATCH_SIZE, 0)

    # padding
    model = SoundNet_UCR(h, classes, channel)
    print(model)
    # 训练
    scores_list= train(train_loader, val_x, val_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, data_set_name, str(seed) + "p", True)
    max_acc = max(scores_list[0])
    print(data_set_name, 'padding accurracy:', max_acc)
    #########################
    # no padding
    model = SoundNet_UCR_np(h, classes, channel)
    print(model)
    # 训练
    scores_list= train(train_loader, val_x, val_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, data_set_name, str(seed) + "np", True)
    max_acc = max(scores_list[0])
    print(data_set_name, 'no padding accurracy:', max_acc)
    #########################
    # small no padding
    model = SoundNet_UCR_np_small(h, classes, channel)
    print(model)
    # 训练
    scores_list= train(train_loader, val_x, val_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, data_set_name, str(seed) + "snp", True)
    max_acc = max(scores_list[0])
    print(data_set_name, 'small no padding accurracy:', max_acc)


def dataset_sp(path_data, data_set_name):
    
    setup_seed(seed)
    # 保存相关
    save_model_path = 'models/sound'
    save_fig_path = 'fig/sound'
    model_name = 'sound_net_sp'
    c2l, l2c = class_label_make(path_data + "TRAIN.arff")

    data_x, data_y = data_read(path_data + "TRAIN.arff", c2l)
    data_x = data_x.astype(np.float32)
    val_x, val_y = data_read(path_data + "TEST.arff", c2l)
    val_x = val_x.astype(np.float32)
    if True in np.isnan(data_x) or True in np.isnan(val_x):
        return
    val_x, val_y = torch.from_numpy(val_x), torch.from_numpy(val_y)

    h = data_x.shape[2]
    channel = data_x.shape[1]
    classes = len(l2c)
    print("Current Dataset:", data_set_name)
    train_loader = get_dataloader(data_x, data_y, BATCH_SIZE, 0)

    # no padding
    model = SoundNet_UCR_np(h, classes, channel)
    print(model)
    # 训练
    scores_list= train(train_loader, val_x, val_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, data_set_name, str(seed) + "np", True)
    max_acc = max(scores_list[0])
    print(data_set_name, 'no padding accurracy:', max_acc)

if __name__ == '__main__':

    # path_data = "./data/Multivariate_arff/StandWalkJump/StandWalkJump_TRAIN.arff"
    # c2l, l2c = class_label_make(path_data)
    # print("Get classes -> label")
    # data_x, data_y = data_read(path_data, c2l)
    # data_x = data_x.astype(np.float32)
    # data_x = torch.from_numpy(data_x)
    # data_y = torch.from_numpy(data_y)
    # print("Get data")
    # k_fold(data_x, data_y, 4, l2c)
    set_list = ["EigenWorms"]
    base_path = "./data/Multivariate_arff/"
    files = os.listdir(base_path)
    files.sort()
    for f in files:
        if f in set_list:
            if os.path.exists(base_path + f + "/" + f + "_" + "TRAIN.arff"):
                dataset_sp(base_path + f + "/" + f + "_", f)


    # main()
    