from experiment import setup_seed, stft_general_data, stft_general_test
from train import train
from utils.dataTrans import get_dataloader
from sklearn.model_selection import train_test_split
from model import Unquantified_2layer_CNN, Unquantified_2layer_CNN_sp, Unquantified_2layer_CNN_sp_test, VGG16_base, VGG16_no_pretrain_padding, VGG16_no_pretrain_no_padding


def VGG_train_test(mode):
    # 参数
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCH = 30
    # 随机种子
    seed = 7266
    setup_seed(seed)

    # 保存相关
    save_model_path = 'models/vgg/stft'
    save_fig_path = 'fig/vgg/stft'
    model_name = 'VGG_base'

    if mode == 1:
        # 数据
        data_x, data_y = stft_general_data()
        train_data_x, dev_data_x, train_data_y, dev_data_y = train_test_split(data_x, data_y, test_size=0.25, shuffle=True, stratify=data_y, random_state=seed)
    else:
        train_data_x, train_data_y = stft_general_data()
        dev_data_x, dev_data_y = stft_general_test()

    # VGG_no_pretrain
    train_loader = get_dataloader(train_data_x, train_data_y, BATCH_SIZE, 0)
    model = VGG16_no_pretrain_no_padding()
    # 训练
    scores_list= train(train_loader, dev_data_x, dev_data_y, model, LR, EPOCH, save_model_path, save_fig_path, model_name, 'no_pretrain_on_padding', seed)
    max_acc = max(scores_list[0])

    print('Accurracy:', max_acc)

if __name__ == '__main__':
    VGG_train_test(1)
    
    