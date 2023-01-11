from model_config import VGG16, AlexNet, LeNet
import numpy as np
import copy
import os
import argparse
from power_sta import net_data, layer_data
import print_line

def delay_cpt(net_structure, stride, pooling_stride):
    delay_list = [1]
    k_idx = 1
    p_idx = 0
    max_delay = stride[0][1]
    for i in range(1, len(net_structure)):
        if net_structure[i] == "c":
            delay_list.append(max_delay)
            max_delay = max_delay * stride[k_idx][1]
            k_idx += 1
        elif net_structure[i] == "p":
            max_delay = max_delay * pooling_stride[p_idx][1]
            p_idx += 1
        else:
            pass
    return delay_list

def find_layer_max(layer_list):
    max_idx = -1
    max_cycle = 0
    for i, l in enumerate(layer_list):
        if l[1][1] > max_cycle:
            max_cycle = l[1][1]
            max_idx = i
    return max_idx

def balance_cpt(out_height_list, in_channel_list, out_channel_list, delay_list):
    pb_num = np.array(out_height_list) * np.array(in_channel_list) * np.array(out_channel_list)
    pre_pb_num = np.ceil(pb_num / np.array(delay_list))
    pb_num = pb_num.tolist()

    min_max_channle = in_channel_list

    # 将每层可能的配置列出
    per_layer_possible_pb_latency = []
    for i, layer in enumerate(pb_num):
        current_layer = []
        for max_channel in range(1, min_max_channle[i] + 1):
            cycle = layer / max_channel
            if cycle - int(cycle) == 0:
                current_layer.append((max_channel, int(cycle)))
        per_layer_possible_pb_latency.append(current_layer)

    # 每一层的最大延迟
    per_layer_max_cycle = [[0, l[0]] for l in per_layer_possible_pb_latency]

    
    # 待测试数据
    all_test_conf = []
    per_all_test_conf = []
    max_cycle_list = []
    max_cycle = 0
    while True:
        # 更改标记
        flag = False
        # 找到当前配置中周期最多的层
        max_layer = find_layer_max(per_layer_max_cycle)
        # 确定该层的配置在该层可能性中的位置
        max_layer_idx = per_layer_max_cycle[max_layer][0]
        # 当前状态的最大周期
        new_max_cycle = per_layer_max_cycle[max_layer][1][1]
        # 将当前配置加入列表
        if new_max_cycle != max_cycle:
            # print(per_layer_max_cycle)
            max_cycle = new_max_cycle

            # 确定其他列的pb数
            new_pre_pb = pre_pb_num / new_max_cycle
            test_new = new_pre_pb[1:] < 1
            if test_new.any():
                pass
            else:
                # last column通道数
                new_pb = np.array(pb_num) / max_cycle
                new_pb = np.ceil(new_pb).astype(int).tolist()
                for i in range(len(new_pb)):
                    if new_pb[i] > min_max_channle[i]:
                        new_pb[i] = min_max_channle[i]
                all_test_conf.append(new_pb)

                # 确定previous column的通道数
                new_pre_pb = np.ceil(new_pre_pb).astype(int).tolist()
                for i in range(len(new_pre_pb)):
                    if new_pre_pb[i] > min_max_channle[i]:
                        new_pre_pb[i] = min_max_channle[i]
                per_all_test_conf.append(new_pre_pb)
                # 最大周期
                max_cycle_list.append(new_max_cycle)

        # 增加该层的pb数并更新
        if max_layer_idx + 1 < len(per_layer_possible_pb_latency[max_layer]):
            per_layer_max_cycle[max_layer] = [max_layer_idx + 1, per_layer_possible_pb_latency[max_layer][max_layer_idx + 1]]
            flag = True
        # 相等说明
        if flag == False:
            break
    return all_test_conf, per_all_test_conf, max_cycle_list

def model_select(m_str):
    if m_str == "vgg":
        return VGG16()
    elif m_str == "alex":
        return AlexNet()
    elif m_str == "le":
        return LeNet()
    else:
        return VGG16()

def model_balance(args):
    path = args.path + "/our/" + args.model
    net = model_select(args.model)

    delay_list = delay_cpt(net.net_structure, net.kernel_stride, net.pooling_kernel_stride)
    # 生成数据
    out_height = net.out_height + [1 for i in range(len(net.fc_in_size))]
    in_channel = net.in_channel + net.fc_in_size
    out_channel = net.out_channel + net.fc_out_size
    delay_list = delay_list + [1 for i in range(len(net.fc_in_size))]
    last_column_channel, per_columns_channel, max_cycle = balance_cpt(out_height, in_channel, out_channel, delay_list)
    
    # 文件写入
    if os.path.exists(path) == False:
        os.makedirs(path)

    print(len(last_column_channel), "pieces of data of our model to be tested have been generated")

    net.net_structure = net.net_structure.replace("cp", "C")
    for i in range(len(last_column_channel)):
        net.last_in_channel = last_column_channel[i][:0 - len(net.fc_in_size)]
        net.pre_in_channel = per_columns_channel[i][:0 - len(net.fc_in_size)]
        net.unified_cycle = max_cycle[i]
        net.multiple = delay_list[:0 - len(net.fc_in_size)]
        net.mini_fc_len = last_column_channel[i][0 - len(net.fc_in_size):]
        config_str = net.basic_str()
        with open(path + "/max_cycle_" + str(max_cycle[i]) + ".txt", "w") as f:
            f.write(config_str)

def base_model_balance(args):
    path = args.path + "/base/" + args.model
    net = model_select(args.model)

    delay_list = delay_cpt(net.net_structure, net.kernel_stride, net.pooling_kernel_stride)
    # 生成数据
    out_height = net.out_height + [1 for i in range(len(net.fc_in_size))]
    in_channel = net.in_channel + net.fc_in_size
    out_channel = net.out_channel + net.fc_out_size
    delay_list = delay_list + [1 for i in range(len(net.fc_in_size))]
    last_column_channel, per_columns_channel, max_cycle = balance_cpt(out_height, in_channel, out_channel, delay_list)
    
    # 文件写入
    if os.path.exists(path) == False:
        os.makedirs(path)

    print(len(last_column_channel), "pieces of data of base model to be tested have been generated")

    net.net_structure = net.net_structure.replace("cp", "C")
    for i in range(len(last_column_channel)):
        net.mini_in_channel = last_column_channel[i][:0 - len(net.fc_in_size)]
        net.unified_cycle = max_cycle[i]
        net.mini_fc_len = last_column_channel[i][0 - len(net.fc_in_size):]
        config_str = net.basic_str()
        with open(path + "/max_cycle_" + str(max_cycle[i]) + ".txt", "w") as f:
            f.write(config_str)

def base_opt_bdw_cpt(mini_in_channel, kernel_size, pooling_kernel_size, mini_fc_len):
    top = 0
    for i in range(len(mini_in_channel)):
        top += mini_in_channel[i] * kernel_size[i][0] * kernel_size[i][1] * 2 + 1
    for i in range(len(mini_fc_len)):
        top += mini_fc_len[i] * 2 + 1

    bandwith = 0
    for i in range(len(mini_in_channel)):
        bandwith += mini_in_channel[i] * kernel_size[i][0] * kernel_size[i][1]
    for i in range(len(pooling_kernel_size)):
        bandwith += pooling_kernel_size[i][0] * pooling_kernel_size[i][1] - 1
    for i in range(len(mini_fc_len)):
        bandwith += mini_fc_len[i]
    return top, bandwith

def our_opt_bdw_cpt(last_in_channel, pre_in_channel, kernel_size, pooling_kernel_size, mini_fc_len):
    top = 0
    for i in range(len(last_in_channel)):
        top += last_in_channel[i] * kernel_size[i][0] * 2 + 2
    for i in range(len(pre_in_channel)):
        top += pre_in_channel[i] * kernel_size[i][0] * (kernel_size[i][1] - 1) * 2 + kernel_size[i][1] - 2
    # 不计算fc
    # for i in range(len(mini_fc_len)):
        # top += mini_fc_len[i] * 2 + 1


    bandwith = 0
    for i in range(len(last_in_channel)):
        bandwith += last_in_channel[i] * kernel_size[i][0] + 1
    for i in range(len(pre_in_channel)):
        bandwith += pre_in_channel[i] * kernel_size[i][0] + kernel_size[i][1] - 1
    for i in range(len(pooling_kernel_size)):
        bandwith += pooling_kernel_size[i][0] * pooling_kernel_size[i][1] - 1
    # 不计算fc
    # for i in range(len(mini_fc_len)):
        # bandwith += mini_fc_len[i]

    return top, bandwith

def base_xy(args):
    net = model_select(args.model)

    delay_list = delay_cpt(net.net_structure, net.kernel_stride, net.pooling_kernel_stride)
    # 生成数据
    out_height = net.out_height + [1 for i in range(len(net.fc_in_size))]
    in_channel = net.in_channel + net.fc_in_size
    out_channel = net.out_channel + net.fc_out_size
    delay_list = delay_list + [1 for i in range(len(net.fc_in_size))]
    last_column_channel, per_columns_channel, max_cycle = balance_cpt(out_height, in_channel, out_channel, delay_list)

    result_list = []
    for i in range(len(last_column_channel)):
        net.mini_in_channel = last_column_channel[i][:0 - len(net.fc_in_size)]
        net.unified_cycle = max_cycle[i]
        net.mini_fc_len = last_column_channel[i][0 - len(net.fc_in_size):]
        tops, bandwith = base_opt_bdw_cpt(net.mini_in_channel, net.kernel_size, net.pooling_kernel_size, net.mini_fc_len)
        result_list.append((net.unified_cycle, tops, bandwith))
    return result_list

def our_xy(args):
    net = model_select(args.model)

    delay_list = delay_cpt(net.net_structure, net.kernel_stride, net.pooling_kernel_stride)
    # 生成数据
    out_height = net.out_height + [1 for i in range(len(net.fc_in_size))]
    in_channel = net.in_channel + net.fc_in_size
    out_channel = net.out_channel + net.fc_out_size
    delay_list = delay_list + [1 for i in range(len(net.fc_in_size))]
    last_column_channel, per_columns_channel, max_cycle = balance_cpt(out_height, in_channel, out_channel, delay_list)

    result_list = []
    for i in range(len(last_column_channel)):
        net.last_in_channel = last_column_channel[i][:0 - len(net.fc_in_size)]
        net.pre_in_channel = per_columns_channel[i][:0 - len(net.fc_in_size)]
        net.unified_cycle = max_cycle[i]
        net.multiple = delay_list[:0 - len(net.fc_in_size)]
        net.mini_fc_len = last_column_channel[i][0 - len(net.fc_in_size):]
        tops, bandwith = our_opt_bdw_cpt(net.last_in_channel, net.pre_in_channel, net.kernel_size, net.pooling_kernel_size, net.mini_fc_len)
        result_list.append((net.unified_cycle, tops, bandwith))
    
    return result_list

def reshape_fig_data(ltb, lp):
    new_a = []
    for a in lp:
        latency = 0
        power = 0
        for s in a:
            if s[0] == "L":
                latency = s[1]
            elif s[0] == "P":
                power = s[1]
        new_a.append((latency, power))
    new_a = sorted(new_a)
    ltb = sorted(ltb)
    # 过滤
    new_ltb = []
    na_c = 0
    for i in range(len(ltb)):
        if ltb[i][0] == new_a[na_c][0]:
            new_ltb.append(ltb[i])
            na_c += 1
        
    ltb = new_ltb
    # latency(cycle), tops, bandwith, power
    ltbp_list = []
    for i in range(len(ltb)):
        l = ltb[i][0]
        t = ltb[i][1]
        b = ltb[i][2]
        p = new_a[i][1]
        ltbp_list.append([l, t, b, p])
    ltbp = np.array(ltbp_list)
    latency = ltbp[:, 0] * 20
    tops = ltbp[:, 1]
    bandwith = ltbp[:, 2]
    power = ltbp[:, 3]
    return latency, tops, bandwith, power

def single_fig_print(args, save_path, mod="base"):
    current = None
    if mod == "base":
        current = base_xy(args)
    else:
        current = our_xy(args)
    a_net = net_data(args.report)
    latency, tops, bandwith, power = reshape_fig_data(current, a_net)
    print_line.fig_plot_latency_x(power, tops, bandwith, latency, save_path)

def double_fig_print(base_args, our_args, save_path):
    current_base = base_xy(base_args)
    current_our = our_xy(our_args)
    a_net_base = net_data(base_args.report)
    a_net_our = net_data(our_args.report)
    base_l, base_t, base_b, base_p = reshape_fig_data(current_base, a_net_base)
    our_l, our_t, our_b, our_p = reshape_fig_data(current_our, a_net_our)
    power_d = (np.array(base_p) - np.array(our_p)) * 1000
    tops_d = np.array(base_t) - np.array(our_t)
    bandwith_d = np.array(base_b) - np.array(our_b)
    print(power_d)
    print(tops_d)
    print(bandwith_d)
    print(our_l)
    print_line.fig_plot_double(base_p, base_t, base_b, our_p, our_t, our_b, our_l, save_path)
    print_line.fig_plot_sig(our_l, power_d, 0, "Power", save_path)
    print_line.fig_plot_sig(our_l, tops_d, 1, "Tops", save_path)
    print_line.fig_plot_sig(our_l, bandwith_d, 2, "Bandwidth", save_path)


def layer_fig_print(base_args, our_args, save_path):
    a_net_base, layer_str = layer_data(base_args.report)
    a_net_base = np.array(sorted(a_net_base))
    a_net_our, _ = layer_data(our_args.report)
    a_net_our = np.array(sorted(a_net_our))

    
    x = a_net_our[:,0]
    print_line.layer_fig_plot(x, a_net_base[:, 1:], a_net_our[:, 1:], layer_str, save_path)
    
def fast_last_cpt(a_net_base, a_net_our):
    fast_cycle = a_net_base[0][0]
    fast_base_layer = np.array(a_net_base[0][1:])
    fast_our_layer = np.array(a_net_our[0][1:])
    fast_persentage = (fast_base_layer - fast_our_layer) / fast_base_layer
    fast = (fast_cycle, fast_persentage)

    last_cycle = a_net_base[-1][0]
    last_base_layer = np.array(a_net_base[-1][1:])
    last_our_layer = np.array(a_net_our[-1][1:])
    last_persentage = (last_base_layer - last_our_layer) / last_base_layer
    last = (last_cycle, last_persentage)

    return fast, last

def persentage_fig_print(base_args, our_args, save_path):
    a_net_base, layer_str = layer_data(base_args.report)
    a_net_base = np.array(sorted(a_net_base))
    a_net_our, _ = layer_data(our_args.report)
    a_net_our = np.array(sorted(a_net_our))

    fast, last = fast_last_cpt(a_net_base, a_net_our)
    print_line.persentage_fig_plot(fast, last, save_path)

def zhu_fig_print(base_args, our_args, save_path, model_name):
    a_net_base, layer_str = layer_data(base_args.report)
    a_net_base = np.array(sorted(a_net_base))
    a_net_our, _ = layer_data(our_args.report)
    a_net_our = np.array(sorted(a_net_our))

    fast, last = fast_last_cpt(a_net_base, a_net_our)
    print_line.zhu_fig_plot(fast, last, save_path, model_name)

def aline_fig_print(base_args, our_args, save_path, model_name):
    a_net_base, layer_str = layer_data(base_args.report)
    a_net_base = np.array(sorted(a_net_base))
    a_net_our, _ = layer_data(our_args.report)
    a_net_our = np.array(sorted(a_net_our))

    
    x = a_net_our[:,0]
    print_line.aline_fig_plot(x, a_net_base[:, 1:], a_net_our[:, 1:], layer_str, save_path, model_name)

def realp_fig_print(base_args, our_args, save_path, model_name, fc=False):
    a_net_base, layer_str = layer_data(base_args.report, fc)
    
    a_net_base = sorted(a_net_base)
    a_net_base = np.array(a_net_base)
    a_net_our, _ = layer_data(our_args.report, fc)
    a_net_our = sorted(a_net_our)
    a_net_our = np.array(a_net_our)
    if model_name == "LeNet":
        idx1 = 0
        idx2 = 0
        for i in range(len(a_net_base)):
            if a_net_base[i][0] == 250:
                idx1 = i
            if a_net_base[i][0] == 315:
                idx2 = i
        a_net_base = np.delete(a_net_base, idx1, axis=0)
        a_net_our = np.delete(a_net_our, idx1, axis=0)

        a_net_base = np.delete(a_net_base, idx2, axis=0)
        a_net_our = np.delete(a_net_our, idx2, axis=0)

    x = a_net_our[:,0]
    print_line.realp_fig_print(x, a_net_base[:, 1:], a_net_our[:, 1:], layer_str, save_path, model_name)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="vgg", help='network')
    parser.add_argument("-p", "--path", type=str, default="./configs", help='save path')
    parser.add_argument("-f", "--function", type=str, default="gen", help='---')
    parser.add_argument("-c", "--config", type=str, default="base", help='base or our or layer or all')
    parser.add_argument("-r", "--report", type=str, default="./configs", help='parents path of report')
    parser.add_argument("-s", "--save_path", type=str, default="./fig", help='fig_save_path')
    args = parser.parse_args()

    if args.function == "gen":
        model_balance(args)
        base_model_balance(args)
    elif args.function == "fig":
        if args.config == "base" or args.config == "our":
            single_fig_print(args, args.save_path + "/" + args.config + "_" + args.model, args.config)
        elif args.config == "all":
            base_args = copy.deepcopy(args)
            base_args.config = "base"
            base_args.report = base_args.report.replace("our", "base")
            base_args.path = base_args.path.replace("our", "base")
            our_args = copy.deepcopy(args)
            our_args.config = "our"
            our_args.report = our_args.report.replace("base", "our")
            our_args.path = our_args.path.replace("base", "our")
            double_fig_print(base_args, our_args, args.save_path + "/" + args.model)
        elif args.config == "layer":
            base_args = copy.deepcopy(args)
            base_args.config = "base"
            base_args.report = base_args.report.replace("our", "base")
            base_args.path = base_args.path.replace("our", "base")
            our_args = copy.deepcopy(args)
            our_args.config = "our"
            our_args.report = our_args.report.replace("base", "our")
            our_args.path = our_args.path.replace("base", "our")
            layer_fig_print(base_args, our_args, args.save_path + "/" + args.model + "_layer")
        
        elif args.config == "persentage":
            base_args = copy.deepcopy(args)
            base_args.config = "base"
            base_args.report = base_args.report.replace("our", "base")
            base_args.path = base_args.path.replace("our", "base")
            our_args = copy.deepcopy(args)
            our_args.config = "our"
            our_args.report = our_args.report.replace("base", "our")
            our_args.path = our_args.path.replace("base", "our")
            persentage_fig_print(base_args, our_args, args.save_path + "/" + args.model + "_persentage")

        elif args.config == "zhu":
            base_args = copy.deepcopy(args)
            base_args.config = "base"
            base_args.report = base_args.report.replace("our", "base")
            base_args.path = base_args.path.replace("our", "base")
            our_args = copy.deepcopy(args)
            our_args.config = "our"
            our_args.report = our_args.report.replace("base", "our")
            our_args.path = our_args.path.replace("base", "our")
            model_name = ""
            if args.model == 'le':
                model_name = "LeNet"
            elif args.model == 'alex':
                model_name = "AlexNet"
            elif args.model == 'vgg':
                model_name = "VGG16"
            zhu_fig_print(base_args, our_args, args.save_path + "/" + args.model + "_zhu", model_name)
            
        elif args.config == "aline":
            base_args = copy.deepcopy(args)
            base_args.config = "base"
            base_args.report = base_args.report.replace("our", "base")
            base_args.path = base_args.path.replace("our", "base")
            our_args = copy.deepcopy(args)
            our_args.config = "our"
            our_args.report = our_args.report.replace("base", "our")
            our_args.path = our_args.path.replace("base", "our")
            model_name = ""
            if args.model == 'le':
                model_name = "LeNet"
            elif args.model == 'alex':
                model_name = "AlexNet"
            elif args.model == 'vgg':
                model_name = "VGG16"
            aline_fig_print(base_args, our_args, args.save_path + "/" + args.model + "_aline", model_name)

        elif args.config == "realp":
            base_args = copy.deepcopy(args)
            base_args.config = "base"
            base_args.report = base_args.report.replace("our", "base")
            base_args.path = base_args.path.replace("our", "base")
            our_args = copy.deepcopy(args)
            our_args.config = "our"
            our_args.report = our_args.report.replace("base", "our")
            our_args.path = our_args.path.replace("base", "our")
            model_name = ""
            if args.model == 'le':
                model_name = "LeNet"
            elif args.model == 'alex':
                model_name = "AlexNet"
            elif args.model == 'vgg':
                model_name = "VGG16"
            realp_fig_print(base_args, our_args, args.save_path + "/" + args.model + "_realp", model_name)
        elif args.config == "realp_fc":
            base_args = copy.deepcopy(args)
            base_args.config = "base"
            base_args.report = base_args.report.replace("our", "base")
            base_args.path = base_args.path.replace("our", "base")
            our_args = copy.deepcopy(args)
            our_args.config = "our"
            our_args.report = our_args.report.replace("base", "our")
            our_args.path = our_args.path.replace("base", "our")
            model_name = ""
            if args.model == 'le':
                model_name = "LeNet"
            elif args.model == 'alex':
                model_name = "AlexNet"
            elif args.model == 'vgg':
                model_name = "VGG16"
            realp_fig_print(base_args, our_args, args.save_path + "/" + args.model + "_realp_fc", model_name, True)
