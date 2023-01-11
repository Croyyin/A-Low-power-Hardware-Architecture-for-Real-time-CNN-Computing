import copy
from operator import mod 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from train import train
from model import Quantified_2layer_CNN, Unquantified_2layer_CNN, VGGMini_CNN
from utils.function import direct_quantize
from utils.dataTrans import get_dataloader
from sklearn.model_selection import train_test_split
from experiment import setup_seed, stft_general_data, stft_general_test

BATCH_SIZE = 64
LR = 0.003
EPOCH = 50

# 参数
in_channels_list_stft = [[2, 32, 32, 64], [2, 2], [2, 4], [2, 8], [2, 16], [2, 32]]
in_channels_list_mfcc = [[1, 1], [1, 2], [1, 4], [1, 8], [1, 16], [1, 32]]
out_channels_list =[[32, 32, 64, 64], [2, 2], [4, 4], [8, 8], [16, 16], [32, 32]]

kernel_size = [3, 3, 3, 3]
stride = [1, 1, 1, 1]

pooling_stride_o = (2, 1)
pooling_stride_n = (2, 2)

feature_map_size_stft = (1025, 173)
feature_map_size_mfcc = (40, 173)

# overlap for stft
def real_train(train_data_x, dev_data_x, train_data_y, dev_data_y, seed_, in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size, save_model_path, save_fig_path, model_name, tag):
    # 确定随机数种子
    setup_seed(seed_)
    train_loader = get_dataloader(train_data_x, train_data_y, BATCH_SIZE)
    # 
    model = VGGMini_CNN(in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size)
    # model = Unquantified_2layer_CNN(in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size)
    # 训练
    scores_list= train(train_loader, dev_data_x, dev_data_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, tag, seed_)
    max_acc = max(scores_list[0])

    return max_acc


# 测试量化位数
def quantization_bit_test(trainloader, test_x, test_y, base_model, max_bits):
    test_logits = base_model(test_x)
    test_predcit = test_logits.argmax(dim=1, keepdim=True)
    acc = accuracy_score(test_y, test_predcit)
    bits_list = [i for i in range(2, max_bits + 1)]
    
    # 未量化前准确率基线
    baseline = [acc for i in range(2, max_bits + 1)]

    # 待量化模型列表
    model_list = [copy.deepcopy(base_model) for i in range(2, max_bits + 1)]
    
    # 量化后准确率
    quantify_acc = []
    for num_bit, model in tqdm(zip(bits_list, model_list)) :
        
        # 量化
        model.quantize(num_bits=num_bit)
        direct_quantize(model, trainloader)
        model.freeze()

        # 计算量化后精确率
        test_logits = model.quantize_inference(test_x)
        test_predcit = test_logits.argmax(dim=1, keepdim=True)
        qacc = accuracy_score(test_y, test_predcit)
        quantify_acc.append(qacc)

    # 画图
    plt.title('Q_Accuracy')
    plt.xlabel('bits')
    plt.ylabel('accuracy')
    plt.plot(bits_list, baseline, 'b', label = 'Baseline')
    plt.plot(bits_list, quantify_acc, 'r', label = 'Quantization accuracy')
    plt.legend()
    plt.savefig('./my_result_stft_1_bits_full.png', bbox_inches='tight')
    plt.show()
    print(quantify_acc)

def run_quantiztion_test():
    # 参数
    in_channels_list = [2, 1]
    out_channels_list =[1, 1]
    kernel_size = [3, 3]
    stride = [1, 1]
    pooling_stride_o = (2, 1)
    feature_map_size = (1025, 173)
    BATCH_SIZE = 64

    # 数据准备
    train_data_x, train_data_y = stft_general_data()
    train_loader = get_dataloader(train_data_x, train_data_y, BATCH_SIZE)
    test_x, test_y = stft_general_test()
    
    # 模型准备
    model_o = Quantified_2layer_CNN(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_o, feature_map_size)
    model_o.load_state_dict(torch.load('models/real/stft/overlap/1/best_model7266.pt'))
    
    # 量化
    quantization_bit_test(train_loader, test_x, test_y, model_o, 32)

# 确定量化参数
def quantization_weight_out(trainloader, test_x, test_y, base_model, num_bit):
    test_logits = base_model(test_x)
    test_predcit = test_logits.argmax(dim=1, keepdim=True)
    real_acc = accuracy_score(test_y, test_predcit)

    # 待量化模型列表
    model = copy.deepcopy(base_model)
    
    # 量化后准确率
    quantify_acc = 0

    # 量化
    model.quantize(num_bits=num_bit)
    direct_quantize(model, trainloader)
    model.freeze()

    # 计算量化后精确率
    test_logits = model.quantize_inference(test_x)
    test_predcit = test_logits.argmax(dim=1, keepdim=True)
    quantify_acc = accuracy_score(test_y, test_predcit)

    print("Accuracy")
    print("Unquantify:", real_acc, "Quantify:", quantify_acc)
    print("Parameters")
    print('Dataset', model.qconv1.qi)
    print('Conv1 weight', model.qconv1.qw)
    print('Conv1 bias', model.qconv1.qb)
    print('Conv2 data', model.qconv1.qo)
    print('Conv2 weight', model.qconv2.qw)
    print('Conv2 bias', model.qconv2.qb)
    print('FC data', model.qconv2.qo)
    print('FC weight', model.qlinear1.qw)
    print('FC bias', model.qlinear1.qb)
    print('Result', model.qlinear1.qo)
    
def run_weight():
    # 参数
    in_channels_list = [2, 1]
    out_channels_list =[1, 1]
    kernel_size = [3, 3]
    stride = [1, 1]
    pooling_stride_o = (2, 1)
    feature_map_size = (1025, 173)
    BATCH_SIZE = 64

    # 数据准备
    train_data_x, train_data_y = stft_general_data()
    train_loader = get_dataloader(train_data_x, train_data_y, BATCH_SIZE)
    test_x, test_y = stft_general_data()

    # 模型
    model_o = Quantified_2layer_CNN(in_channels_list, out_channels_list, kernel_size, stride,pooling_stride_o, feature_map_size)
    model_o.load_state_dict(torch.load('models/real/stft/overlap/1/best_model7266.pt'))

    # 
    quantization_weight_out(train_loader, test_x, test_y, model_o, 8)


def test_main(mode):
    seed = 7266
    i = 0
    # 保存相关
    save_model_path = 'models/very/stft'
    save_fig_path = 'fig/very/stft'

    model_name_o = 'VGGMini'
    tag = out_channels_list[i][0]
    if mode == 1:
        # 数据
        data_x, data_y = stft_general_data()
        train_data_x, dev_data_x, train_data_y, dev_data_y = train_test_split(data_x, data_y, test_size=0.25, shuffle=True, stratify=data_y, random_state=seed)
    else:
        train_data_x, train_data_y = stft_general_data()
        dev_data_x, dev_data_y = stft_general_test()


    feature_map_size = feature_map_size_stft
    print('对于STFT数据集')

    max_acc = real_train(train_data_x, dev_data_x, train_data_y, dev_data_y, seed, in_channels_list_stft[i], out_channels_list[i], kernel_size, stride, pooling_stride_n, feature_map_size, save_model_path, save_fig_path, model_name_o, tag)
    print('Accurracy:', max_acc)



if __name__ == '__main__':
    # run_weight()
    test_main(1)
    # run_quantiztion_test()


