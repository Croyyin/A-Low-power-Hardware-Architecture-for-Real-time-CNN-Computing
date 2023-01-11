import os
from scipy.ndimage.measurements import label
from tqdm import tqdm
import torch.nn as nn

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils.dataTrans import mkdir

def train(train_loader, dev_data_x, dev_data_y, model, lr, epoch, save_model_path, save_fig_path, model_name, tag, bef, only_acc=False):
    # 图
    color_list = ['cornflowerblue', 'tomato', 'gold', 'springgreen']
    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({"font.size": 13})
    ax = fig.gca()

    # 指标
    scores_name_list = ['accuracy', 'precision', 'recall', 'f1']
    scores_list = []
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    # 统计损失
    train_loss = []
    epoch_list = []

    train_acc = []
    data_num = 0

    # 统计验证集指标
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    best_f1 = 0
    for e in tqdm(range(epoch), desc='Training'):
        epoch_list.append(e)
        t_loss = 0
        
        # 训练
        model.train()
        train_acc_count = 0
        data_num = 0
        for b_x, b_y in train_loader:
            output = model(b_x)

            loss = loss_function(output, b_y.long())  

            optimizer.zero_grad()           
            loss.backward()                 
            optimizer.step()   
            
            t_loss += loss.item()
            train_pred_y = torch.max(output, 1)[1]
            train_acc_count += (train_pred_y == b_y).sum().item()
            data_num += b_y.shape[0]

        train_acc.append(float(train_acc_count) / float(data_num))

        t_loss = t_loss / len(train_loader)
        scheduler.step(t_loss)
        train_loss.append(t_loss)

        # loss 图
        plt.title('Loss line chart')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        ax.plot(epoch_list, train_loss, c=color_list[0], label = 'Training loss')
        plt.legend()
        
        # 路径创建
        loss_path = os.path.join(save_fig_path, model_name, tag)
        mkdir(loss_path)

        plt.savefig(os.path.join(loss_path, 'train_loss' + bef + '.svg'), bbox_inches='tight')
        plt.cla()

        # 在验证集上看指标
        model.eval()
        with torch.no_grad():
            dev_logits = model(dev_data_x)
            dev_predict = torch.max(dev_logits, 1)[1]
    
            # 计算指标
            accuracy_list.append(accuracy_score(dev_data_y, dev_predict))
            precision_list.append(precision_score(dev_data_y, dev_predict, average='macro', zero_division=0))
            recall_list.append(recall_score(dev_data_y, dev_predict, average='macro', zero_division=0))
            f1 = f1_score(dev_data_y, dev_predict, average='macro', zero_division=0)
            f1_list.append(f1)
    
            if f1 > best_f1:
                #保存
                model_path = os.path.join(save_model_path, model_name, tag)
                mkdir(model_path)
                torch.save(model.state_dict(), os.path.join(model_path, 'best_model' + bef + '.pt'))
                best_f1 = f1
            
    scores_list.append(accuracy_list)
    scores_list.append(precision_list)
    scores_list.append(recall_list)
    scores_list.append(f1_list)

    if only_acc:
        plt.title('Scores line chart')
        plt.xlabel('epoch')
        plt.ylabel('rate')
        ax.plot(epoch_list, accuracy_list, c=color_list[0], label="Accuracy")
    else:
        c_f = 0
        for name, sc in zip(scores_name_list, scores_list):
            plt.title('Scores line chart')
            plt.xlabel('epoch')
            plt.ylabel('rate')
            ax.plot(epoch_list, sc, c=color_list[c_f], label=name)
            c_f += 1

    ax.plot(epoch_list, train_acc, c='darkorchid', label='Train acc')

    plt.legend()

    score_path = os.path.join(save_fig_path, model_name, tag)
    mkdir(score_path)

    plt.savefig(os.path.join(score_path, 'scores' + bef + '.svg'), bbox_inches='tight')
    plt.cla()

    return scores_list

