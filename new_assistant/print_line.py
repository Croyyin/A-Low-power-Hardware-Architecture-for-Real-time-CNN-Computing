from logging import handlers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick
import math
color_list = ['royalblue', 'orange', 'limegreen', 'crimson', 'mediumorchid', 'dimgray', 'mediumpurple', 'chocolate', 'lime', 'lightpink', 'maroon', 'navy', 'tan']

def find_first_pos(list_1):
    i = -1
    for j, l in enumerate(list_1):
        if l > 0:
            i = j
            break
    return i

def fig_plot(power_y, latency_y, bandwidth_y, operations_x, save_path):
    fig = plt.figure(figsize=(8, 3))
    plt.rcParams.update({"font.size": 13})
    ax1 = fig.gca()

    plt.title('Power and Latency')
    # left
    ax1.set_xlabel('operations')
    ax1.set_ylabel('Power (mW)')
    l1, = ax1.plot(operations_x, power_y, c=color_list[0], label = 'Power', marker='^')

    # Right
    ax2 = ax1.twinx()
    ax2.set_ylabel('Latency (us)')
    l2, = ax2.plot(operations_x, latency_y, c=color_list[1], label = 'Latency', marker='o')

    # Right 2
    ax3 = ax1.twinx()
    ax3.set_ylabel('Bandwidth (GBps)')
    l3, = ax3.plot(operations_x, bandwidth_y, c=color_list[2], label = 'Bandwidth', marker='*')

    ax1.tick_params(axis='y', colors=color_list[0])
    ax2.tick_params(axis='y', colors=color_list[1])
    ax3.tick_params(axis='y', colors=color_list[2])
    ax1.spines['right'].set_visible(False)
    ax3.spines['left'].set_color(color_list[0])
    ax2.spines['right'].set_color(color_list[1])
    ax3.spines['right'].set_color(color_list[2])
    ax3.spines['right'].set_position(('axes',1.2))
    ax3.set_ylim(0)
    ax1.set_ylim(0)

    ax1.yaxis.label.set_color(color_list[0])
    ax2.yaxis.label.set_color(color_list[1])
    ax3.yaxis.label.set_color(color_list[2])


    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    plt.legend(handles=[l1, l2, l3], labels=['Power', 'Latency', 'Bandwidth'])
    plt.savefig(save_path + '.svg')

def fig_plot_latency_x(power_y, operations_y, bandwidth_y, latency_x, save_path):
    fig = plt.figure(figsize=(18, 13))
    plt.rcParams.update({"font.size": 13})
    ax1 = fig.gca()
    plt.title('Power and Latency')
    
    # 文本输出
    for i in range(len(power_y)):
        topsw = (operations_y[i] / 1000) / (20 * power_y[i])
        print("latency: ", latency_x[i], "TOPs/W: ", power_y[i]);

    # left
    ax1.set_xlabel('latency')
    ax1.set_ylabel('Power (mW)')
    l1, = ax1.plot(latency_x, power_y, c=color_list[0], label = 'Power', marker='^')

    # Right
    ax2 = ax1.twinx()
    ax2.set_ylabel('Operations')
    l2, = ax2.plot(latency_x, operations_y, c=color_list[1], label = 'Operations', marker='o')

    # Right 2
    ax3 = ax1.twinx()
    ax3.set_ylabel('Bandwidth (GBps)')
    l3, = ax3.plot(latency_x, bandwidth_y, c=color_list[2], label = 'Bandwidth', marker='*')

    ax1.tick_params(axis='y', colors=color_list[0])
    ax2.tick_params(axis='y', colors=color_list[1])
    ax3.tick_params(axis='y', colors=color_list[2])
    ax1.spines['right'].set_visible(False)
    ax3.spines['left'].set_color(color_list[0])
    ax2.spines['right'].set_color(color_list[1])
    ax3.spines['right'].set_color(color_list[2])
    ax3.spines['right'].set_position(('axes',1.2))
    ax3.set_ylim(0)
    ax1.set_ylim(0)

    ax1.yaxis.label.set_color(color_list[0])
    ax2.yaxis.label.set_color(color_list[1])
    ax3.yaxis.label.set_color(color_list[2])


    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    plt.legend(handles=[l1, l2, l3], labels=['Power', 'Operations', 'Bandwidth'])
    plt.savefig(save_path + '.svg')

def fig_plot_double(power_yb, operations_yb, bandwidth_yb, power_yo, operations_yo, bandwidth_yo, latency_x, save_path):
    fig = plt.figure(figsize=(8, 3))
    plt.rcParams.update({"font.size": 13})
    ax1 = fig.gca()

    plt.title('Power and Latency')
    # left
    ax1.set_xlabel('latency')
    ax1.set_ylabel('Power (mW)')
    l1_b, = ax1.plot(latency_x, power_yb, c=color_list[0], label = 'Power base', marker='^')
    l1_o, = ax1.plot(latency_x, power_yo, c=color_list[0], label = 'Power our', marker='o')
    # Right
    ax2 = ax1.twinx()
    ax2.set_ylabel('Operations')
    l2_b, = ax2.plot(latency_x, operations_yb, c=color_list[1], label = 'Operations base', marker='^')
    l2_o, = ax2.plot(latency_x, operations_yo, c=color_list[1], label = 'Operations our', marker='o')
    # Right 2
    ax3 = ax1.twinx()
    ax3.set_ylabel('Bandwidth (GBps)')
    l3_b, = ax3.plot(latency_x, bandwidth_yb, c=color_list[2], label = 'Bandwidth base', marker='^')
    l3_o, = ax3.plot(latency_x, bandwidth_yo, c=color_list[2], label = 'Bandwidth our', marker='o')
    ax1.tick_params(axis='y', colors=color_list[0])
    ax2.tick_params(axis='y', colors=color_list[1])
    ax3.tick_params(axis='y', colors=color_list[2])
    ax1.spines['right'].set_visible(False)
    ax3.spines['left'].set_color(color_list[0])
    ax2.spines['right'].set_color(color_list[1])
    ax3.spines['right'].set_color(color_list[2])
    ax3.spines['right'].set_position(('axes',1.2))
    ax3.set_ylim(0)
    ax1.set_ylim(0)

    ax1.yaxis.label.set_color(color_list[0])
    ax2.yaxis.label.set_color(color_list[1])
    ax3.yaxis.label.set_color(color_list[2])


    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    plt.legend(handles=[l1_b, l1_o, l2_b, l2_o, l3_b, l3_o], labels=['Power base', 'Power our', 'Operations base', 'Operations our', 'Bandwidth base', 'Bandwidth our'])
    plt.savefig(save_path + '.svg')

def fig_plot_div(power_y, operations_y, bandwidth_y, latency_x, save_path):
    fig = plt.figure(figsize=(8, 3))
    plt.rcParams.update({"font.size": 13})
    ax1 = fig.gca()

    plt.title('Power and Latency')
    # left
    ax1.set_xlabel('latency')
    ax1.set_ylabel('Power (mW)')
    l1, = ax1.plot(latency_x, power_y, c=color_list[0], label = 'Power', marker='^')

    # Right
    ax2 = ax1.twinx()
    ax2.set_ylabel('Operations')
    l2, = ax2.plot(latency_x, operations_y, c=color_list[1], label = 'Operations', marker='o')

    # Right 2
    ax3 = ax1.twinx()
    ax3.set_ylabel('Bandwidth (GBps)')
    l3, = ax3.plot(latency_x, bandwidth_y, c=color_list[2], label = 'Bandwidth', marker='*')

    ax1.tick_params(axis='y', colors=color_list[0])
    ax2.tick_params(axis='y', colors=color_list[1])
    ax3.tick_params(axis='y', colors=color_list[2])
    ax1.spines['right'].set_visible(False)
    ax3.spines['left'].set_color(color_list[0])
    ax2.spines['right'].set_color(color_list[1])
    ax3.spines['right'].set_color(color_list[2])
    ax3.spines['right'].set_position(('axes',1.2))


    ax1.yaxis.label.set_color(color_list[0])
    ax2.yaxis.label.set_color(color_list[1])
    ax3.yaxis.label.set_color(color_list[2])


    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    plt.legend(handles=[l1, l2, l3], labels=['Power', 'Operations', 'Bandwidth'])
    plt.savefig(save_path + '.svg')

def fig_plot_sig(x, y, idx, label, save_path):
    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({"font.size": 13})
    ax1 = fig.gca()
    
    ax1.plot(x, y, color_list[idx], label=label)

    # left
    ax1.set_xlabel('latency')
    ax1.set_ylabel(label)
    ax1.legend()
    plt.savefig(save_path + "_" + label + '.svg')

def layer_fig_plot(x, y1_list, y2_list, layer_str, save_path):
    fig = plt.figure(figsize=(40, 15))
    plt.rcParams.update({"font.size": 13})
    
    num = y1_list.shape[1] + 1
    partial_num = math.ceil(num / 2)

    div_list = y1_list - y2_list

    for i in range(num):
        plt.subplot(2, partial_num, i + 1)
        if i != num - 1:
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
            plt.xlabel("Latency")
            plt.ylabel("Power")
            plt.plot(x, (y1_list[:, i] - y2_list[:, i]) / y1_list[:, i], c=color_list[i], marker="^", label="Persentage")
            plt.title(layer_str[i])
            plt.legend()
        else:
            plt.xlabel("Latency")
            plt.ylabel("Difference in power")
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
            for d in range(div_list.shape[1]):
                plt.plot(x, div_list[:, d], c=color_list[d], label=layer_str[d])
            plt.title("All layer")
            plt.legend()
    plt.savefig(save_path + '.svg')

def persentage_fig_plot(fast, last, save_path):
    fig = plt.figure(figsize=(15, 8))
    plt.rcParams.update({"font.size": 13})

    y1 = fast[1]
    y2 = last[1]
    x = np.array([i + 1 for i in range(y1.shape[0])])
    plt.xlabel("Layer")
    plt.ylabel("Power reduce persentage")
    plt.plot(x, y1, c=color_list[0], marker="^", label="Latency: " + str(fast[0]))
    plt.plot(x, y2, c=color_list[1], marker="o", label="Latency: " + str(last[0]))
    plt.title("Persentage")
    plt.legend()
    plt.savefig(save_path + '.svg')

def zhu_fig_plot(fast, last, save_path, model_name):
    fig = plt.figure(figsize=(15, 8))
    plt.rcParams.update({"font.size": 38})

    y1 = fast[1]
    y1_pos = find_first_pos(y1)
    y2 = last[1]
    y2_pos = find_first_pos(y2)
    x = np.array([i + 1 for i in range(y1.shape[0])])

    lgg = 5.0 / (4 * len(y2))


    total_width, n = 1 - lgg, 2
    width = total_width / n
    # x = x - (total_width - width) / 2

    # plt.bar(x, y1,  width=width, fc=color_list[0], label="Latency: " + str(fast[0]))
    # plt.bar(x + width, y2, width=width, fc=color_list[1], label="Latency: " + str(last[0]))
    # plt.xticks(np.array([i + 1 for i in range(y1.shape[0])]))
    # plt.xlabel("Convolutional layer index")
    # plt.ylabel("Power consumption reduction rate")
    # plt.title(model_name)
    # plt.legend()
    # plt.savefig(save_path + '.svg')
    plt.bar(x[0:y1_pos], y1[0:y1_pos], fc="w", width=width, edgecolor=color_list[0], label="Negative rate", hatch="////")
    plt.bar(x[y1_pos:], y1[y1_pos:],  width=width, fc=color_list[0], label="Positive rate")
    plt.plot(x, y1, c="black", marker="o", label="Trendline")
    plt.xlabel("Convolutional layer depth")
    plt.ylabel("Power reduction rate")
    # plt.title(model_name + ": " + "latency =" + str(fast[0]))
    # plt.legend(loc=4)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.savefig(save_path + "_" + str(fast[0]) + '.svg')
    plt.cla()

    plt.bar(x[0:y2_pos], y2[0:y2_pos], fc="w", width=width, edgecolor=color_list[1], label="Negative rate", hatch="////")
    plt.bar(x[y2_pos:], y2[y2_pos:], width=width, fc=color_list[1], label="Positive rate")
    plt.plot(x, y2, c="black", marker="o", label="Trendline")
    plt.xlabel("Convolutional layer depth")
    plt.ylabel("Power reduction rate")
    # plt.title(model_name + ": " + "latency =" + str(last[0]))
    # plt.legend(loc=4)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.savefig(save_path + "_" + str(last[0]) + '.svg')
    plt.cla()

def aline_fig_plot(x, y1_list, y2_list, layer_str, save_path, model_name):
    fig = plt.figure(figsize=(15, 8))
    plt.rcParams.update({"font.size": 25})
    
    maker_list = ["^", "o", "*", "+"]

    num = y1_list.shape[1]

    div_list = y1_list - y2_list
    plt.xlabel("Latency")
    plt.ylabel("Power consumption reduction rate")
    plt.title(model_name)

    loop_list = []
    if model_name == "VGG16":
        loop_list = [2, 2, 3, 3, 3]
    elif model_name == "AlexNet":
        loop_list = [1, 1, 3]
    elif model_name == "LeNet":
        loop_list = [1, 1]

    i = 0
    for l, loop in enumerate(loop_list):
        for k in range(loop):
            plt.plot(x, (y1_list[:, i] - y2_list[:, i]) / y1_list[:, i], c=color_list[l], marker=maker_list[k], label="Conv. layer " + str(i + 1))
            i += 1
        
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    plt.subplots_adjust(left=0.12, right=0.7, top=0.9, bottom=0.12)
    plt.legend(loc=(1.03, -0.05)) 
    plt.savefig(save_path + '.svg')

def realp_fig_print(x, y1_list, y2_list, layer_str, save_path, model_name):
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update({"font.size": 25})
    
    maker_list = ["^", "o", "*", "+"]

    num = y1_list.shape[1]

    all_y1 = np.array([np.sum(line) for line in y1_list]) * 1000
    all_y2 = np.array([np.sum(line) for line in y2_list]) * 1000

    div_list = all_y1 - all_y2
    x = (20 * x) / 1000

    lgg = 5.0 / (4 * len(y2_list))

    new_x = [ x[i] for i in range(1, len(x))]
    d_x = x[:-1]
    min_x = np.min(new_x - d_x)
    width = min_x * 0.7

    ax1 = fig.gca()

    max_rate = np.max(div_list / all_y1)
    min_rate = np.min(div_list / all_y1)
    print(max_rate)
    print(min_rate)
    
    if model_name == "VGG16":
        min_y1 = -250
        max_y1 = 450
        t_y1 = [50, 150, 250, 350]

        min_y2 = 0.3
        max_y2 = 0.7
        t_y2 = [0.3, 0.35, 0.4, 0.45, 0.5]
    elif model_name == "AlexNet":
        min_y1 = -25
        max_y1 = 70
        t_y1 = [15, 25, 35, 45, 55]

        min_y2 = 0.2
        max_y2 = 0.8
        t_y2 = [0.2, 0.3, 0.4, 0.5]
    elif model_name == "LeNet":
        min_y1 = 3
        max_y1 = 7.5
        t_y1 = [5, 5.5, 6, 6.5, 7]

        min_y2 = -0.06
        max_y2 = 0.20
        t_y2 = [-0.06, -0.02, 0.02, 0.06, 0.1]
        ax1.set_xlim(3, 10)
        ax1.set_xticks([3, 4, 5, 6, 7, 8, 9, 10])


    
    # left
    ax1.set_xlabel('Latency (us)')
    ax1.set_ylabel('Power (mW)')
    l1, = ax1.plot(x, all_y1, c=color_list[0], marker=maker_list[0], label="Conventional design")
    l11, = ax1.plot(x, all_y2, c=color_list[3], marker=maker_list[0], label="Optimized design")
    ax1.set_ylim(min_y1, max_y1)
    ax1.set_yticks(t_y1)

    # Right
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reduction rate')
    ax2.set_ylim(min_y2, max_y2)
    ax2.set_yticks(t_y2)
    # l2, = ax2.plot(x, div_list / all_y1, c=color_list[1], marker=maker_list[1], label="Power consumption reduction rate")
    l2 = ax2.bar(x, div_list / all_y1,  width=width, fc=color_list[1], label="Reduction rate")
    # ax1.tick_params(axis='y', colors=color_list[0])
    # ax2.tick_params(axis='y', colors=color_list[1])
    ax1.spines['right'].set_visible(False)
    # ax2.spines['right'].set_color(color_list[1])
    # ax1.spines['bottom'].set_position(('data',0))#这个位置的括号要注意
    # ax1.yaxis.label.set_color(color_list[0])
    # ax2.yaxis.label.set_color(color_list[1])


    plt.subplots_adjust(left=0.15, right=0.8, top=0.85, bottom=0.15)
    plt.legend(handles=[l1, l11, l2], labels=['Conventional design', 'Optimized design', 'Reduction rate'])
    plt.savefig(save_path + '.svg')
