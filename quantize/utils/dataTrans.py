import os
import re
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
# audio
import librosa
import librosa.display
import sklearn
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

# 读取采样
def load_clip(filename):
    x, sr = librosa.load(filename, sr=22050)
    if 4 * sr - x.shape[0] > 0: 
        x = np.pad(x,(0,4 * sr - x.shape[0]),'constant')
    else:
        x = x[:4 * sr]
    return x, sr

# MFCC特征提取
def extract_feature_mfcc(filename):
    x, sr = load_clip(filename)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    mfccs = mfccs.astype(float)
    mfcc = sklearn.preprocessing.scale(mfccs, axis=0)
    return mfcc

# STFT特征提取
def extract_feature_stft(filename):
    x, sr = load_clip(filename)
    stft = librosa.stft(x)

    stft_abs = np.abs(stft).astype(float)
    stft_angle = np.angle(stft).astype(float)

    nstft_abs = sklearn.preprocessing.scale(stft_abs, axis=0).reshape(1, 1025, 173)
    nstft_angle = sklearn.preprocessing.scale(stft_angle, axis=0).reshape(1, 1025, 173)

    return nstft_abs, nstft_angle

# MFCC数据集
def load_dataset_mfcc(filenames):
    features, labels = np.empty((0,40,173)), np.empty(0)
    for filename in tqdm(filenames):
        mfccs = extract_feature_mfcc(filename)

        features = np.append(features,mfccs[None],axis=0)
        labels = np.append(labels, filename.split('/')[-1].split('-')[1])
    return np.array(features), np.array(labels, dtype=np.int)

def load_dataset_one_dim(filenames):
    features, labels = np.empty((0, 1, 88200, 1)), np.empty(0)

    for filename in tqdm(filenames):

        data, sr = load_clip(filename)
        features = np.append(features, data.reshape(1, 1, 88200, 1), axis=0)
        labels = np.append(labels, filename.split('/')[-1].split('-')[1])

    return np.array(features), np.array(labels, dtype=np.int)

# STFT数据集
def load_dataset_stft(filenames):
    features, labels = np.empty((0, 2, 1025, 173)), np.empty(0)

    for filename in tqdm(filenames):
        one_sample = np.empty((0, 1025, 173))

        stft_real, stft_abv = extract_feature_stft(filename)

        one_sample = np.append(one_sample, stft_real, axis=0)
        one_sample = np.append(one_sample, stft_abv, axis=0)
        
        features = np.append(features, one_sample.reshape(1, 2, 1025, 173), axis=0)
        labels = np.append(labels, filename.split('/')[-1].split('-')[1])

    return np.array(features), np.array(labels, dtype=np.int)    

# 获取dataloader
def get_dataloader(data_x, data_y, batch_size, workers=1):
    mydataset = Mydataset(data_x, data_y)
    dataloader = DataLoader(dataset=mydataset, shuffle=True, batch_size=batch_size, num_workers=workers)

    return dataloader

def get_data(x_path, y_path):
    # 从文件读取
    data_x = pickle.load(open(x_path, 'rb')).astype(np.float32)
    data_y = pickle.load(open(y_path, 'rb')).astype(np.float32)

    # reshape
    data_x = data_x.reshape((data_x.shape[0],1,data_x.shape[1],data_x.shape[2]))

    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    
    return data_x, data_y

class Mydataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.label = labels

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

def mkdir(path):
    # 判断路径是否存在
    isExists=os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

if __name__ == "__main__":
    # y, _ = load_clip("./data/UrbanSound8K/audio/fold1/31840-3-0-0.wav")
    
    x = np.arange(0, np.pi * 100, 0.1)
    y_rand = np.random.random(len(x))
    y = np.sin(x) + 0.6 * np.cos(2*x) + np.cos(0.3 * x) + np.sin(x + 2) + y_rand
    fig = plt.figure(figsize=(20, 3))
    plt.plot(x, y, c='cornflowerblue')
    plt.savefig('./fig.svg', bbox_inches='tight')