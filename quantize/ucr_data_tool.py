from scipy.io import arff
import numpy as np

class2idx = {"standing": 0, "walking": 1, "jumping": 2}


def data_look(path):
    data, meta = arff.loadarff(path)
    print(data.shape, len(data))
    print(data[0][0].shape, len(data[0][0]))
    print(data[0][0][0].shape, len(data[0][0][0]))

def class_label_make(path):
    data, meta = arff.loadarff(path)
    class_list = []
    for i in range(len(data)):
        class_list.append(data[i][1].decode("UTF-8"))
    class_list = list(set(class_list))
    label_list = [i for i in range(len(class_list))]

    class2label = {}
    label2class = {}

    for i in range(len(class_list)):
        class2label[class_list[i]] = label_list[i]
        label2class[label_list[i]] = class_list[i]
    return class2label, label2class
    

def data_read(path, class2idx):
    data, meta = arff.loadarff(path)
    channel_num = len(data[0][0])
    column_num = len(data[0][0][0])
    items, labels = np.empty((0, channel_num, column_num, 1)), np.empty(0)
    for i in range(len(data)):
        label = class2idx[data[i][1].decode('UTF-8')]
        item = np.array(data[i][0].tolist()).reshape(1, channel_num, column_num, 1)
        labels = np.append(labels, label)
        items = np.append(items, item, axis=0)

    return np.array(items), np.array(labels, dtype=np.int)

if __name__ == "__main__":
    # data_x, data_y = data_read("./data/StandWalkJump/StandWalkJump_TEST.arff", class2idx)
    # print(data_x.shape)
    # print(data_y.shape)
    path = "./data/Multivariate_arff/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.arff"
    l1, l2 = class_label_make(path)
    data_look(path)  
    print(len(l1))