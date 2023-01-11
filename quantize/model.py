import torch.nn as nn
import torch.nn.functional as F
from math import floor
import torchvision.models as models

from utils.quantify import *

class Unquantified_2layer_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size):
        super(Unquantified_2layer_CNN, self).__init__()
        
        # 计算全连接层数据个数
        linear_in_row = feature_map_size[0]
        linear_in_col = feature_map_size[1]
        for i in range(2):
            linear_in_row = ((linear_in_row - kernel_size[i]) / stride[i]) + 1
            linear_in_row = floor((linear_in_row - 2) / pooling_stride[0]) + 1
            linear_in_col = ((linear_in_col - kernel_size[i]) / stride[i]) + 1
            linear_in_col = floor((linear_in_col - 2) / pooling_stride[1]) + 1
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=0),                              
            nn.ReLU(),                     
            nn.MaxPool2d(kernel_size=2, stride=pooling_stride),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1], stride=stride[1], padding=0),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=pooling_stride),                
        )
        self.linear1 = nn.Linear(linear_in_row * out_channels[1] * linear_in_col, 10)
        self.out = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(self.linear1(x))
        return output

class Unquantified_2layer_CNN_sp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size):
        super(Unquantified_2layer_CNN_sp, self).__init__()
        
        # 计算全连接层数据个数
        linear_in_row = feature_map_size[0]
        linear_in_col = feature_map_size[1]
        for i in range(2):
            linear_in_row = ((linear_in_row - kernel_size[i]) / stride[i]) + 1
            linear_in_row = floor((linear_in_row - 2) / pooling_stride[0]) + 1
            linear_in_col = ((linear_in_col - kernel_size[i]) / stride[i]) + 1
            linear_in_col = floor((linear_in_col - 2) / pooling_stride[1]) + 1
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=0),                              
            nn.ReLU(),                     
            nn.MaxPool2d(kernel_size=2, stride=pooling_stride),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1], stride=stride[1], padding=0),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=pooling_stride),                
        )
        self.linear1 = nn.Linear(linear_in_row * out_channels[1] * linear_in_col, 10)
        self.out = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(self.linear1(x))
        return output

    def fc_forward(self, x, previous_result=None, previous_first_column=None, mode=1):
        x = self.conv1(x)
        x = self.conv2(x)
        linear_in_row = x.size(2)
        linear_in_column = x.size(3)
        x = x.view(x.size(0), -1)

        # 获取第一列和最后一列
        current_first_column_index = [i * linear_in_column for i in range(linear_in_row)]
        current_last_column_index = [(i + 1) * linear_in_column - 1 for i in range(linear_in_row)]
        current_first_column = x[:, current_first_column_index]
        current_last_column = x[:, current_last_column_index]

        # None
        current_result_wout = None
        previous_result_wout = None

        # 累计模式
        if mode == 0:
            # 获取第一列和最后一列的参数
            first_column_weight = self.linear1.weight.data[:, current_first_column_index]
            last_column_weight = self.linear1.weight.data[:, current_last_column_index]
            fc_bias = torch.zeros([10])
            # 分别计算第一列和最后一列的全连接
            previous_first = F.linear(previous_first_column, first_column_weight, fc_bias)
            current_last = F.linear(current_last_column, last_column_weight, fc_bias)
            # 求和
            before_softmax = previous_result - previous_first + current_last

            current_result = self.linear1(x)
            current_result_wout = current_result - current_last
            previous_result_wout = previous_result - previous_first

            # print(current_result_wout, previous_result_wout)

        else: # 正常模式
            before_softmax = self.linear1(x)
            
        output = self.out(before_softmax)

        return output, before_softmax, current_first_column

class Unquantified_2layer_CNN_sp_test(nn.Module):
    l_in_row = 0
    l_in_col = 0
    f_col_w = None
    l_col_w = None
    def __init__(self, in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size):
        super(Unquantified_2layer_CNN_sp_test, self).__init__()
        
        # 计算全连接层数据个数
        linear_in_row = feature_map_size[0]
        linear_in_col = feature_map_size[1]

        for i in range(2):
            linear_in_row = ((linear_in_row - kernel_size[i]) / stride[i]) + 1
            linear_in_row = floor((linear_in_row - 2) / pooling_stride[0]) + 1
            linear_in_col = ((linear_in_col - kernel_size[i]) / stride[i]) + 1
            linear_in_col = floor((linear_in_col - 2) / pooling_stride[1]) + 1
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=0),                              
            nn.ReLU(),                     
            nn.MaxPool2d(kernel_size=2, stride=pooling_stride),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1], stride=stride[1], padding=0),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=pooling_stride),                
        )
        self.linear1 = nn.Linear(linear_in_row * out_channels[1] * linear_in_col, 10)
        self.out = nn.Softmax(dim=1)
        
        self.l_in_row = linear_in_row
        self.l_in_col = linear_in_col

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(self.linear1(x))
        return output

    def fc_forward(self, x, previous_result=None, previous_first_column=None, mode=1):
        x = self.conv1(x)
        x = self.conv2(x)
        linear_in_row = x.size(2)
        linear_in_column = x.size(3)
        x = x.view(x.size(0), -1)

        # 获取第一列和最后一列
        current_first_column_index = [i * linear_in_column for i in range(linear_in_row)]
        current_last_column_index = [(i + 1) * linear_in_column - 1 for i in range(linear_in_row)]
        current_first_column = x[:, current_first_column_index]
        current_last_column = x[:, current_last_column_index]

        # None
        current_result_wout = None
        previous_result_wout = None

        # 累计模式
        if mode == 0:
            # 获取第一列和最后一列的参数
            first_column_weight = self.linear1.weight.data[:, current_first_column_index]
            last_column_weight = self.linear1.weight.data[:, current_last_column_index]

            if torch.equal(first_column_weight, self.f_col_w) and torch.equal(last_column_weight, self.l_col_w):
                print('Yest')

            fc_bias = torch.zeros([10])
            # 分别计算第一列和最后一列的全连接
            previous_first = F.linear(previous_first_column, first_column_weight, fc_bias)
            current_last = F.linear(current_last_column, last_column_weight, fc_bias)
            # 求和
            before_softmax = previous_result - previous_first + current_last

            current_result = self.linear1(x)
            current_result_wout = current_result - current_last
            previous_result_wout = previous_result - previous_first

            # print(current_result_wout, previous_result_wout)

        else: # 正常模式
            before_softmax = self.linear1(x)
            
        output = self.out(before_softmax)

        return output, before_softmax, current_first_column

    def test_parameter_set(self):
        new_tensor = (torch.rand(10, self.l_in_row, self.l_in_col) * 10).floor()
        print(new_tensor)
        print('---')
        self.f_col_w = new_tensor[:, :, 0]
        self.l_col_w = new_tensor[:, :, -1]
        self.linear1.weight.data = new_tensor.view(10, -1)
        self.linear1.bias.data = self.linear1.bias.data * 0
        print(self.f_col_w)
        print('---')
        print(self.l_col_w)
        print('---')


class VGGMini_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size):
        super(VGGMini_CNN, self).__init__()
        
        # 计算全连接层数据个数
        linear_in_row = feature_map_size[0]
        linear_in_col = feature_map_size[1]
        for i in range(4):
            if i % 2 == 0:
                linear_in_row = ((linear_in_row - kernel_size[i]) / stride[i]) + 1
                linear_in_col = ((linear_in_col - kernel_size[i]) / stride[i]) + 1
            else:
                linear_in_row = ((linear_in_row - kernel_size[i]) / stride[i]) + 1
                linear_in_row = floor((linear_in_row - 2) / pooling_stride[0]) + 1
                linear_in_col = ((linear_in_col - kernel_size[i]) / stride[i]) + 1
                linear_in_col = floor((linear_in_col - 2) / pooling_stride[1]) + 1
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=0),                              
            nn.ReLU(),                     
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1], stride=stride[1], padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=pooling_stride),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2], stride=stride[2], padding=0),     
            nn.ReLU(),                      
            nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_size[3], stride=stride[3], padding=0),     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=pooling_stride),                
        )
        self.linear1 = nn.Linear(linear_in_row * out_channels[3] * linear_in_col, 100)
        self.linear2 = nn.Linear(100, 10)
        self.out = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        output = self.out(self.linear2(x))
        return output

class Quantified_2layer_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pooling_stride, feature_map_size):

        # 计算全连接层数据个数
        linear_in_row = feature_map_size[0]
        linear_in_col = feature_map_size[1]
        for i in range(2):
            linear_in_row = ((linear_in_row - kernel_size[i]) / stride[i]) + 1
            linear_in_row = floor((linear_in_row - 2) / pooling_stride[0]) + 1
            linear_in_col = ((linear_in_col - kernel_size[i]) / stride[i]) + 1
            linear_in_col = floor((linear_in_col - 2) / pooling_stride[1]) + 1
        
        self.pooling_stride = pooling_stride
        super(Quantified_2layer_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0], stride=stride[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1], stride=stride[1], padding=0)  
        self.linear1 = nn.Linear(linear_in_row * linear_in_col * out_channels[1], 10)
        self.out = nn.Softmax(dim=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, self.pooling_stride)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, self.pooling_stride)

        x = x.view(x.size(0), -1)
        output = self.out(self.linear1(x))
        return output

    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU(num_bits=num_bits)
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=self.pooling_stride, padding=0, num_bits=num_bits)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU(num_bits=num_bits)
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=self.pooling_stride, padding=0, num_bits=num_bits)
        self.qlinear1 = QLinear(self.linear1, qi=False, qo=True, num_bits=num_bits)

    # 前向计算，统计scale和zero_point
    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qrelu1(x)
        
        x = self.qconv2(x)
        x = self.qmaxpool2d_2(x)
        x = self.qrelu2(x)
        
        x = x.view(x.size(0), -1)
        output = self.out(self.qlinear1(x))
        return output

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qlinear1.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)

        qx = qx.view(qx.size(0), -1)
        qx = self.qlinear1.quantize_inference(qx)
        qx = self.qlinear1.qo.dequantize_tensor(qx)
        output = self.out(qx)
        return output

class CNN_2layer_1channel_1channel_3k_1linear(nn.Module):
    def __init__(self):
        super(CNN_2layer_1channel_1channel_3k_1linear, self).__init__()

        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),                     
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),                
        )
        self.linear1 = nn.Linear(10414, 10)
        self.out = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(self.linear1(x))
        return output

class CNN_2layer_1channel_1channel_3k_2linear(nn.Module):
    def __init__(self):
        super(CNN_2layer_1channel_1channel_3k_2linear, self).__init__()

        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),                     
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),                
        )
        self.linear1 = nn.Linear(10414, 500)
        self.linear2 = nn.Linear(500, 10)
        self.out = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        output = self.out(self.linear2(x))
        return output

class VGG16_base(nn.Module):
    def __init__(self):
        super(VGG16_base, self).__init__()
        pretrained_net = models.vgg16(pretrained=True)
        self.prefix = nn.Conv2d(2, 3, 3, 1, padding=1)
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[0][:17]) # 第一段
        self.stage2 = nn.Sequential(*list(pretrained_net.children())[0][17:24]) # 第二段
        self.stage3 = nn.Sequential(*list(pretrained_net.children())[0][24:31]) # 第三段

        self.linear1 = nn.Linear(5 * 32 * 512, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 10)
        self.out = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.prefix(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.out(x)
        return x

class VGG16_no_pretrain_padding(nn.Module):
    def __init__(self):
        super(VGG16_no_pretrain_padding, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),        
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),   
                         
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),        
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),   
                         
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),        
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),   
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),        
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),        
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),   
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),                              
            nn.ReLU(),          
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.linear1 = nn.Linear(10 * 64 * 512, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 10)
        self.out = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.out(x)
        return x

class VGG16_no_pretrain_no_padding(nn.Module):
    def __init__(self):
        super(VGG16_no_pretrain_no_padding, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),        
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),   
                         
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),        
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),   
                         
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),        
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),   
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),        
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),        
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),   
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0),                              
            nn.ReLU(),          
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.linear1 = nn.Linear(58 * 5 * 512, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 10)
        self.out = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.out(x)
        return x


class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        self.conv8 = nn.Conv2d(1024, 10, kernel_size=(4, 1),
                                    stride=(2, 1))
        self.out = nn.Softmax(dim=1)
    def forward(self, waveform):
        batch_size = waveform.shape[0]
        x = self.conv1(waveform)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        
        x = self.out(x.view(batch_size, -1))
        return x

class SoundNet_no_padding(nn.Module):
    def __init__(self):
        super(SoundNet_no_padding, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        self.conv8 = nn.Conv2d(1024, 10, kernel_size=(4, 1),
                                    stride=(2, 1))
        self.out = nn.Softmax(dim=1)
    def forward(self, waveform):
        batch_size = waveform.shape[0]
        x = self.conv1(waveform)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        print(x.shape)
        exit()
        x = self.conv7(x)
        x = self.relu7(x)
        
        x = self.conv8(x)
        
        x = self.out(x.view(batch_size, -1))
        return x



class SoundNet_UCR(nn.Module):
    def __init__(self, h, classes, in_channel):
        super(SoundNet_UCR, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        h = int((h - 64 + 2 * 32) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(16, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))
        h = int((h - 8) / 8 + 1)
        if h < 32:
            self.conv8 = nn.Conv2d(16, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        h = int((h - 32 + 2 * 16) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(32, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))
        h = int((h - 8) / 8 + 1)
        if h < 16:
            self.conv8 = nn.Conv2d(32, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)
        h = int((h - 16 + 2 * 8) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(64, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)
        h = int((h - 8 + 2 * 4) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(128, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        h = int((h - 4 + 2 * 2) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(256, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))
        h = int((h - 4) / 4 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(256, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)
        h = int((h - 4 + 2 * 2) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(512, classes, kernel_size=(h, 1), stride=(2, 1))
            return
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)
        h = int((h - 4 + 2 * 2) / 2 + 1)
        self.conv8 = nn.Conv2d(1024, classes, kernel_size=(h, 1), stride=(2, 1))
    def forward(self, waveform):
        batch_size = waveform.shape[0]
        x = self.conv1(waveform)
        x = self.relu1(x)
        if hasattr(self, "maxpool1"):
            x = self.maxpool1(x)
        if hasattr(self, "conv2"):
            x = self.conv2(x)
            x = self.relu2(x)
        if hasattr(self, "maxpool2"):
            x = self.maxpool2(x)
        if hasattr(self, "conv3"):
            x = self.conv3(x)
            x = self.relu3(x)
        if hasattr(self, "conv4"):
            x = self.conv4(x)
            x = self.relu4(x)
        if hasattr(self, "conv5"):
            x = self.conv5(x)
            x = self.relu5(x)
        if hasattr(self, "maxpool5"):
            x = self.maxpool5(x)
        if hasattr(self, "conv6"):
            x = self.conv6(x)
            x = self.relu6(x)
        if hasattr(self, "conv7"):
            x = self.conv7(x)
            x = self.relu7(x)

        x = self.conv8(x)
        x = F.softmax(x.view(batch_size, -1), dim=1)
        return x

class SoundNet_UCR_np(nn.Module):
    def __init__(self, h, classes, in_channel):
        super(SoundNet_UCR_np, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=(64, 1), stride=(2, 1))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        h = int((h - 64) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(16, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))
        h = int((h - 8) / 8 + 1)
        if h < 32:
            self.conv8 = nn.Conv2d(16, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        h = int((h - 32) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(32, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))
        h = int((h - 8) / 8 + 1)
        if h < 16:
            self.conv8 = nn.Conv2d(32, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)
        h = int((h - 16) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(64, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)
        h = int((h - 8) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(128, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        h = int((h - 4) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(256, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))
        h = int((h - 4) / 4 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(256, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)
        h = int((h - 4) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(512, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)
        h = int((h - 4) / 2 + 1)
        self.conv8 = nn.Conv2d(1024, classes * 10, kernel_size=(h, 1), stride=(2, 1))
        self.linear = nn.Linear(classes * 10, classes)
    def forward(self, waveform):
        batch_size = waveform.shape[0]
        x = self.conv1(waveform)
        x = self.relu1(x)
        if hasattr(self, "maxpool1"):
            x = self.maxpool1(x)
        if hasattr(self, "conv2"):
            x = self.conv2(x)
            x = self.relu2(x)
        if hasattr(self, "maxpool2"):
            x = self.maxpool2(x)
        if hasattr(self, "conv3"):
            x = self.conv3(x)
            x = self.relu3(x)
        if hasattr(self, "conv4"):
            x = self.conv4(x)
            x = self.relu4(x)
        if hasattr(self, "conv5"):
            x = self.conv5(x)
            x = self.relu5(x)
        if hasattr(self, "maxpool5"):
            x = self.maxpool5(x)
        if hasattr(self, "conv6"):
            x = self.conv6(x)
            x = self.relu6(x)
        if hasattr(self, "conv7"):
            x = self.conv7(x)
            x = self.relu7(x)

        x = self.conv8(x)
        x = self.linear(x.view(batch_size, -1))
        x = F.softmax(x, dim=1)
        return x

class SoundNet_UCR_np_small(nn.Module):
    def __init__(self, h, classes, in_channel):
        super(SoundNet_UCR_np_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=(16, 1), stride=(2, 1))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        h = int((h - 16) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(16, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))
        h = int((h - 8) / 8 + 1)
        if h < 32:
            self.conv8 = nn.Conv2d(16, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(16, 1), stride=(2, 1))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        h = int((h - 16) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(32, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))
        h = int((h - 8) / 8 + 1)
        if h < 16:
            self.conv8 = nn.Conv2d(32, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)
        h = int((h - 16) / 2 + 1)
        if h < 8:
            self.conv8 = nn.Conv2d(64, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)
        h = int((h - 8) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(128, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        h = int((h - 4) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(256, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))
        h = int((h - 4) / 4 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(256, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)
        h = int((h - 4) / 2 + 1)
        if h < 4:
            self.conv8 = nn.Conv2d(512, classes * 10, kernel_size=(h, 1), stride=(2, 1))
            self.linear = nn.Linear(classes * 10, classes)
            return
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)
        h = int((h - 4) / 2 + 1)
        self.conv8 = nn.Conv2d(1024, classes * 10, kernel_size=(h, 1), stride=(2, 1))
        self.linear = nn.Linear(classes * 10, classes)
    def forward(self, waveform):
        batch_size = waveform.shape[0]
        x = self.conv1(waveform)
        x = self.relu1(x)
        if hasattr(self, "maxpool1"):
            x = self.maxpool1(x)
        if hasattr(self, "conv2"):
            x = self.conv2(x)
            x = self.relu2(x)
        if hasattr(self, "maxpool2"):
            x = self.maxpool2(x)
        if hasattr(self, "conv3"):
            x = self.conv3(x)
            x = self.relu3(x)
        if hasattr(self, "conv4"):
            x = self.conv4(x)
            x = self.relu4(x)
        if hasattr(self, "conv5"):
            x = self.conv5(x)
            x = self.relu5(x)
        if hasattr(self, "maxpool5"):
            x = self.maxpool5(x)
        if hasattr(self, "conv6"):
            x = self.conv6(x)
            x = self.relu6(x)
        if hasattr(self, "conv7"):
            x = self.conv7(x)
            x = self.relu7(x)

        x = self.conv8(x)
        x = self.linear(x.view(batch_size, -1))
        x = F.softmax(x, dim=1)
        return x


class SoundNet_ERing(nn.Module):
    def __init__(self, h, classes, in_channel):
        super(SoundNet_ERing, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=(64, 1), stride=(2, 1))
        self.relu1 = nn.ReLU(True)
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.linear = nn.Linear(16, classes)
    def forward(self, waveform):
        batch_size = waveform.shape[0]
        x = self.conv1(waveform)
        x = self.relu1(x)
        x = self.linear(x.view(batch_size, -1))
        x = F.softmax(x, dim=1)
        return x

