import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.function import FakeQuantize
from math import e, pow

# 计算S和零点
def calc_ScaleZeroPoint(min_val, max_val, num_bits=8):
    n = num_bits - 1
    
    qmin = - 2 ** n
    qmax = 2. ** n - 1.

    # S=(rmax-rmin)/(qmax-qmin)
    scale = float((max_val - min_val) / (qmax - qmin))
    
    # Z=round(qmax-rmax/scale)
    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax
    
    zero_point = int(zero_point)

    return scale, zero_point

# 量化张量
def quantize_tensor(x, scale, zero_point, num_bits=8, signed=True):

    # 有符号数
    if signed: 
        qmin = 0. - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    
    # 无符号数
    else:      
        qmin = 0.
        qmax = 2.**num_bits - 1.
 
    # q=round(r/S+Z)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()     
    
    return q_x.float()  

# 反量化
def dequantize_tensor(q_x, scale, zero_point):
    # r=S(q-Z)
    return scale * (q_x - zero_point)


class QParam:

    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scale = None
        self.zero_point = None
        self.min = None
        self.max = None

    def update(self, tensor):
        if self.max is None or self.max < tensor.max():
            self.max = tensor.max()
        
        if self.min is None or self.min > tensor.min():
            self.min = tensor.min()
        
        self.scale, self.zero_point = calc_ScaleZeroPoint(self.min, self.max, self.num_bits)
    
    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def __str__(self):
        info = 'scale: %.10f ' % self.scale
        info += 'zp: %d ' % self.zero_point
        info += 'min: %.6f ' % self.min
        info += 'max: %.6f' % self.max
        return info

# 量化模型
class QModule(nn.Module):

    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')

# 量化卷积
class QConv2d(QModule):

    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=num_bits)

        # 更新
        self.qw.update(self.conv_module.weight.data)
        self.qb.update(self.conv_module.bias.data)


    def freeze(self, qi=None, qo=None):

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M1 = self.qw.scale * self.qi.scale / self.qo.scale
        self.M2 = self.qb.scale / self.qo.scale

        # 量化 + 零点
        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.qweight_data = self.conv_module.weight.data
        self.conv_module.weight.data = self.M1 * (self.conv_module.weight.data - self.qw.zero_point)

        self.conv_module.bias.data = self.qb.quantize_tensor(self.conv_module.bias.data)
        self.qbias_data = self.conv_module.bias.data
        self.conv_module.bias.data = self.M2 * (self.conv_module.bias.data - self.qb.zero_point)

    def forward(self, x):

        # 更新 量化输入：qi
        if hasattr(self, 'qi'):
            self.qi.update(x)
        x = self.conv_module(x)
        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x

    # 量化推理
    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x.round_() 
        x = x + self.qo.zero_point    
        x.clamp_(0. - 2. ** (self.num_bits - 1), 2. ** (self.num_bits - 1) - 1.  ).round_()
        return x

class QLinear(QModule):

    def __init__(self, linear_module, qi=True, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.linear_module = linear_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=num_bits)

        # 更新
        self.qw.update(self.linear_module.weight.data)
        self.qb.update(self.linear_module.bias.data)

    def freeze(self, qi=None, qo=None):

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M1 = self.qw.scale * self.qi.scale / self.qo.scale
        self.M2 = self.qb.scale / self.qo.scale

        self.linear_module.weight.data = self.qw.quantize_tensor(self.linear_module.weight.data)
        self.qweight_data = self.linear_module.weight.data
        self.linear_module.weight.data = self.M1 * (self.linear_module.weight.data - self.qw.zero_point)

        self.linear_module.bias.data = self.qb.quantize_tensor(self.linear_module.bias.data)
        self.qbias_data = self.linear_module.bias.data
        self.linear_module.bias.data = self.M2 * (self.linear_module.bias.data - self.qb.zero_point)

    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)
        x = self.linear_module(x)
        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.linear_module(x)
        x.round_() 
        x = x + self.qo.zero_point
        x.clamp_(0. - 2. ** (self.num_bits - 1), 2. ** (self.num_bits - 1) - 1.  ).round_()
        return x


class QReLU(QModule):

    def __init__(self, qi=True, num_bits=None):
        super(QReLU, self).__init__(qi=qi, num_bits=num_bits)

    def freeze(self, qi=None):
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
        x = F.relu(x)

        return x
    
    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x

class QMaxPooling2d(QModule):

    def __init__(self, kernel_size=3, stride=1, padding=0, qi=True, num_bits=None):
        super(QMaxPooling2d, self).__init__(qi=qi, num_bits=num_bits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def freeze(self, qi=None):
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

        return x

    def quantize_inference(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

