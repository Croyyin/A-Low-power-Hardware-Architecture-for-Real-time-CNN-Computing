import os
import re
import argparse
import copy

def statistics_module(path, fc=False):
    files = os.listdir(path)
    layers = []
    for f in files:
        if fc:
            if "power" in f:
                with open(path + "/" + f, 'r') as rpt:
                    lines = rpt.readlines()
                    for line in lines:
                        if "Total Dynamic Power" in line:
                            power_str = re.findall(r"\d+\.?\d* u?m?W", line) 
                            layers.append((f[:-10], power_str))
        else:
            if "FC" not in f and "power" in f:
                with open(path + "/" + f, 'r') as rpt:
                    lines = rpt.readlines()
                    for line in lines:
                        if "Total Dynamic Power" in line:
                            power_str = re.findall(r"\d+\.?\d* u?m?W", line) 
                            layers.append((f[:-10], power_str))
    return layers

def standardization(layers_power):
    new_layers = []
    for l in layers_power:
        head_str = l[0][0] + "_" + l[0].split("_")[-1]
        power_num = float(l[1][0].split(" ")[0])
        power_unit = l[1][0].split(" ")[1]
        if power_unit[0] == "u":
            power_num *= 1e-6
        elif power_unit[0] == "m":
            power_num *= 1e-3
        else:
            pass
        new_layers.append((head_str, power_num))
    new_layers = sorted(new_layers)
    end = 0
    new_c_layer = []
    for i, l in enumerate(new_layers):
        if "C" in l[0]:
            end = i
            new_c_layer.append((int(l[0].replace("C_", "")), l[1]))
    new_c_layer = sorted(new_c_layer)

    new_cc_layer = []
    for i, l in enumerate(new_c_layer):
        new_cc_layer.append(("C_" + str(l[0]), l[1]))

    new_layers = new_cc_layer + new_layers[end + 1:]
    return new_layers

def total_power(layers_power):
    power = 0
    for l in layers_power:
        power += l[1]
    return power

def net_data(path, fc=False):
    dirs = os.listdir(path)
    a_net = []
    for d in dirs:
        if os.path.isdir(path + "/" + d):
            layers = statistics_module(path + "/" + d + "/rpt", fc)
            layers = standardization(layers)
            power = total_power(layers)
            latency = int(d.split("_")[-1])
            layers.append(("L", latency))
            layers.append(("P", power))
            a_net.append(layers)

    return a_net

def layer_str_re(layer_str):
    bc_layer = copy.deepcopy(layer_str)
    for i in range(len(bc_layer)):
        if "C" in bc_layer[i]:
            bc_layer[i] = bc_layer[i].replace("C_", "Convolutional layer ")
        elif "F" in bc_layer[i]:
            bc_layer[i] = bc_layer[i].replace("F_", "Fully connected layer ")
    return bc_layer

def layer_data(path, fc=False):
    dirs = os.listdir(path)
    net_list = []
    layer_list = []
    for d in dirs:
        a_net = []
        a_layer = []
        if os.path.isdir(path + "/" + d):
            layers = statistics_module(path + "/" + d + "/rpt", fc)
            layers = standardization(layers)
            latency = int(d.split("_")[-1])
            a_net.append(latency)
            for l in layers:
                a_net.append(l[1])
                a_layer.append(l[0])     
            net_list.append(a_net)
            layer_list.append(a_layer)
    return net_list, layer_str_re(layer_list[0])

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="./configs", help='parent dir of rpt')
    args = parser.parse_args()

    a_net = net_data(args.path)
    print(a_net)
    