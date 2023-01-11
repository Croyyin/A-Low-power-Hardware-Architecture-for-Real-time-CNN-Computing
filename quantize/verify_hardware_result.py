from os import path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from utils.quantization import *

class test_CNN(nn.Module):
    def __init__(self, weight1, bias1, weight2, bias2, weightfc, biasfc):
        super(test_CNN, self).__init__()
        self.weight1 = nn.Parameter(weight1)
        self.bias1 = nn.Parameter(bias1)
        self.weight2 = nn.Parameter(weight2)
        self.bias2 = nn.Parameter(bias2)

        self.weightfc = nn.Parameter(weightfc)
        self.biasfc = nn.Parameter(biasfc)
    
        
    def forward(self, x):
        x = F.conv2d(x, weight=self.weight1, bias=self.bias1, stride=1, padding=0)
        x = F.max_pool2d(x, 2, (2, 2))
        x = F.conv2d(x, self.weight2, self.bias2, stride=1, padding=0)
        x = F.max_pool2d(x, 2, (2, 2))
        x = x.view(x.size(0), -1)
        # output = x
        output = F.linear(x, self.weightfc, self.biasfc)
        return output

def read_data(path, shape_tuple):
    with open(path, 'r') as f:
        lines = f.readlines()
        d_list = [int(i[0]) for i in lines]
        
        data = torch.tensor(d_list).reshape(shape_tuple).float()
        return data
 

def main():
    data = read_data(path='./resources/data.txt', shape_tuple=(1, 2, 15, 14))
    
    weight1 = read_data('./resources/weight1st.txt', (1, 2, 3, 3))
    
    bias1 = read_data('./resources/bias1st.txt', (1))
    weight2 = read_data('./resources/weight2nd.txt', (1, 1, 3, 3))
    bias2 = read_data('./resources/bias2nd.txt', (1))
    weightfc =read_data('./resources/weightfc.txt', (4, 14))
    
    biasfc = read_data('./resources/biasfc.txt', (4))
    weightfc_none =read_data('./resources/weightfc_none.txt', (4, 4))

    model = test_CNN(weight1, bias1, weight2, bias2, weightfc_none, biasfc)

    out = model(data)
    print(out)

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
    d =  cpt_h_r('./logtest.txt', 2, 2, 7)
    print(d.shape)
    print(d)


 
