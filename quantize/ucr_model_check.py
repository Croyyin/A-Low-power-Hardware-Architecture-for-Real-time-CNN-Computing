import ucr_sound_net
from experiment import setup_seed, one_dim_data
from train import train
from utils.dataTrans import get_dataloader
from sklearn.model_selection import train_test_split
from model import SoundNet, SoundNet_no_padding, SoundNet_UCR,SoundNet_ERing, SoundNet_UCR_np
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from ucr_data_tool import class_label_make, data_read
import os
# 参数
BATCH_SIZE = 8
LR = 1e-3
EPOCH = 200
# 随机种子
seed = 7266

def main(path_data, data_set_name):
    print("Current Dataset:", data_set_name)
    setup_seed(seed)
    # 保存相关
    save_model_path = 'models/sound'
    save_fig_path = 'fig/sound'
    model_name = 'sound_BM'
    c2l, l2c = class_label_make(path_data + "TRAIN.arff")

    data_x, data_y = data_read(path_data + "TRAIN.arff", c2l)
    data_x = data_x.astype(np.float32)
    val_x, val_y = data_read(path_data + "TEST.arff", c2l)
    val_x = val_x.astype(np.float32)
    val_x, val_y = torch.from_numpy(val_x), torch.from_numpy(val_y)

    h = data_x.shape[2]
    channel = data_x.shape[1]
    classes = len(l2c)

    train_loader = get_dataloader(data_x, data_y, BATCH_SIZE, 0)
    model = SoundNet_UCR_np(h, classes, channel)
    print(model)
    # 训练
    scores_list= train(train_loader, val_x, val_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, data_set_name, str(seed), True)
    max_acc = max(scores_list[0])
    print(data_set_name, 'accurracy:', max_acc)


if __name__ == '__main__':

    data_set_list = ["BasicMotions"]
    base_path = "./data/Multivariate_arff/"
    files = os.listdir(base_path)
    files.sort()
    for f in files:
        if f in data_set_list:
            if os.path.exists(base_path + f + "/" + f + "_" + "TRAIN.arff"):
                main(base_path + f + "/" + f + "_", f)


    # main()