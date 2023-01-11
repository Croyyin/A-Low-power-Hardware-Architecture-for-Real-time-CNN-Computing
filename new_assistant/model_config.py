from config_save import str_gen

def height_list_cpt(net_structure, in_height, kernel_size, stride, padding, pooling_size, pooling_stride, is_in=False):
    result = []
    k_idx = 0
    p_idx = 0
    if is_in:
        result.append(in_height)
    current = in_height
    for i in range(len(net_structure)):
        if net_structure[i] == "c":
            current = int((current + 2 * padding[k_idx][0] - kernel_size[k_idx][0]) / stride[k_idx][0]) + 1
            result.append(current)
            k_idx += 1
        elif net_structure[i] == "p":
            current = int((current - pooling_size[p_idx][0]) / pooling_stride[p_idx][0]) + 1
            p_idx += 1
        else:
            pass
    if is_in:
        result = result[:-1]
    return result




class VGG16(str_gen):
    def __init__(self) -> None:
        super().__init__()
        self.net_structure = "ccpccpcccpcccpcccpfff"
        self.input_size = [224, 224]
        
        self.in_channel = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
        self.out_channel = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

        self.kernel_size = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        self.kernel_stride = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        self.padding = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]

        self.pooling_kernel_size= [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
        self.pooling_kernel_stride = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        self.in_height = height_list_cpt(self.net_structure, self.input_size[0], self.kernel_size, self.kernel_stride, self.padding, self.pooling_kernel_size, self.pooling_kernel_stride,True)
        self.out_height = height_list_cpt(self.net_structure, self.input_size[0], self.kernel_size, self.kernel_stride, self.padding, self.pooling_kernel_size, self.pooling_kernel_stride, False)
        
        self.fc_in_size = [25088, 4096, 4096]
        self.fc_out_size = [4096, 4096, 21]

class AlexNet(str_gen):
    def __init__(self) -> None:
        super().__init__()
        self.net_structure = "cpcpcccpfff"
        self.input_size = [224, 224]
        
        self.in_channel = [3, 96, 256, 384, 384]
        self.out_channel = [96, 256, 384, 384, 256]

        self.kernel_size = [[11, 11], [5, 5], [3, 3], [3, 3], [3, 3]]
        self.kernel_stride = [[4, 4], [1, 1], [1, 1], [1, 1], [1, 1]]
        self.padding = [[2, 2], [2, 2], [1, 1], [1, 1], [1, 1]]

        self.pooling_kernel_size= [[3, 3], [3, 3], [3, 3]]
        self.pooling_kernel_stride = [[2, 2], [2, 2], [2, 2]]

        self.in_height = height_list_cpt(self.net_structure, self.input_size[0], self.kernel_size, self.kernel_stride, self.padding, self.pooling_kernel_size, self.pooling_kernel_stride,True)
        self.out_height = height_list_cpt(self.net_structure, self.input_size[0], self.kernel_size, self.kernel_stride, self.padding, self.pooling_kernel_size, self.pooling_kernel_stride, False)
        
        self.fc_in_size = [9216, 4096, 4096]
        self.fc_out_size = [4096, 4096, 21]

class LeNet(str_gen):
    def __init__(self) -> None:
        super().__init__()
        self.net_structure = "cpcpfff"
        self.input_size = [32, 32]
        
        self.in_channel = [1, 6]
        self.out_channel = [6, 16]

        self.kernel_size = [[5, 5], [5, 5]]
        self.kernel_stride = [[1, 1], [1, 1]]
        self.padding = [[0, 0], [0, 0]]

        self.pooling_kernel_size= [[2, 2], [2, 2]]
        self.pooling_kernel_stride = [[2, 2], [2, 2]]

        self.in_height = height_list_cpt(self.net_structure, self.input_size[0], self.kernel_size, self.kernel_stride, self.padding, self.pooling_kernel_size, self.pooling_kernel_stride,True)
        self.out_height = height_list_cpt(self.net_structure, self.input_size[0], self.kernel_size, self.kernel_stride, self.padding, self.pooling_kernel_size, self.pooling_kernel_stride, False)
        
        self.fc_in_size = [400, 120, 84]
        self.fc_out_size = [120, 84, 10]