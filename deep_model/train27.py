# 直接在原来的整个数据集上训练
# 然后在测试集上进行测试
from collections import OrderedDict
from operator import mod

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch.nn as nn
import torch
import time
import os
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from torch.optim import lr_scheduler

from densenet_plus import Net4
from test import extra_feats_pipeline
from __init__ import global_var


try:
    from PIL import Image
except ImportError:
    print(ImportError)
    import Image

device = "cuda:0"

class TrainDataset(Dataset):
    def __init__(self, train_img_path, train_img_label, data_transforms=None):
        self.train_img_path = train_img_path
        self.train_img_label = train_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.train_img_path)

    def __getitem__(self, item):
        cur_img_path = self.train_img_path[item]
        cur_img_label = self.train_img_label[item]

        cur_img_data = load_img_data_from_path(cur_img_path)
        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, cur_img_data, cur_img_label


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_img_data_from_path(img_path, loader=pil_loader):
    return loader(img_path)


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def cmp_img(x, y):
    x_base = os.path.basename(x)
    y_base = os.path.basename(y)
    return int(x_base[:3]) - int(y_base[:3])


def get_number(num):
    if num[1:4].isdigit():
        return int(num[1:4])
    elif num[1:3].isdigit():
        return int(num[1:3])
    elif num[1:2].isdigit():
        return int(num[1:2])
    else:
        return -1


def get_available_data_by_order(data_dir):
    train_img_path = []
    train_img_label = []

    img_names_list = sorted(os.listdir(data_dir), key=lambda x: get_number(x))
    img_names_list = sorted(img_names_list, key=lambda x:x[0])

    for img_name in img_names_list:
        img_path = os.path.join(data_dir, img_name)
        # label = 0 if img_name.startswith('B') else 1
        label = 0 if 'B' in img_name or 'b' in img_name else 1
        train_img_path.append(img_path)
        train_img_label.append(label)
        # print(img_path)
        # print(label)

    # print(len(train_img_path))
    # print(train_img_label)
    # print(train_img_path)
    return train_img_path, train_img_label


def get_consistent_mean(img_paths, img_labels):
    img_labels = [1 if 'M' in img_path else 0 for img_path in img_paths]

    consistent_mean_sum = torch.zeros(3)
    consistent_std_sum = torch.zeros(3)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    for img_path in img_paths:
        img_data = pil_loader(img_path)
        img_data = data_transforms(img_data)
        # print(type(img_data))
        # print(img_data.size())
        for i in range(3):
            consistent_mean_sum[i] += img_data[i, :, :].mean()
            consistent_std_sum[i] += img_data[i, :, :].std()

    return consistent_mean_sum/len(img_paths), consistent_std_sum/len(img_paths)


class Dataset(Dataset):
    def __init__(self, img_path, img_label, data_transforms=None):
        self.img_path = img_path
        self.img_label = img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        cur_img_path = self.img_path[item]
        cur_img_label = self.img_label[item]

        print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)
        extra_feats = extra_feats_pipeline(cur_img_data)
        print('1 in test dataset class extra feats shape:', extra_feats.shape)
        # print(type(extra_feats))
        # print((torch.Tensor(extra_feats) == torch.Tensor([0])).all().item())
        if (torch.Tensor(extra_feats) == torch.Tensor([0])).all().item():
            print('No extra feats')
            extra_feats = torch.Tensor(extra_feats)
        else:
            print("Extra Feats!")
            extra_feats = np.asarray(extra_feats)
            extra_feats = (extra_feats - np.mean(extra_feats)) / np.std(extra_feats)
            extra_feats = torch.Tensor(extra_feats)
        print('2 in test dataset class extra feats shape:', extra_feats.shape)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, cur_img_data, cur_img_label, extra_feats


def train_model(net, train_paths, train_labels, test_paths, test_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                model_name='', img_feats_dict=dict(), pkl_file='',
                ):
    """
    :param net:
    :param train_paths:
    :param train_labels:
    :param test_paths:
    :param test_labels:
    :param num_epochs:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param use_gpu:
    :param pkl_file:
    :param model_name:
    :param img_feats_dict:
    :return:
    """
    # 进行训练和验证
    # 这里还是每一次训练就进行一次对应的验证吧  如果每间隔一定数量的训练次数再验证可能会丢失最好的结果
    # 需要编写的内容有:
    #  1.每次训练的时候把参数训练结果都保存起来
    #  2.记录验证效果最好的那次所对应的相关信息，包括迭代次数、最好的验证准确率、此时模型的参数
    #  3.训练过程无非是
    #  0）设定当前状态为训练状态，优化器的梯度要清0
    #  1）获取批数据;把批数据放在对应的设备上；
    #  2）获取模型对该数据的输出
    #  3）计算模型的损失和准确率
    #  4）对损失进行反向传播，学习率调整器调整学习率的变化
    #  4.验证过程与训练过程基本一致，有两点不同：
    #  0）当前状态设置为不训练
    #  1）不进行反向传播和学习率调整
    print('*' * 25)
    global test_acc, test_loss, train_acc, train_loss

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        start_epoch = int(os.path.basename(pkl_file).split('_')[-2])

        print('the start epoch is:', start_epoch)

    else:
        print('the start epoch is:', start_epoch)
        # return

    model_pkls_name = 'pkls_' + model_name + '_' + str(num_epochs)

    print("model_pkls_name:", model_pkls_name)

    dir_path = global_var.base_deep_prefix + '/' + model_name + '/'

    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)
            print('make succ:', dir_path)
        except Exception as e:
            print(e)

    best_acc = 0.0
    best_model_wts = net.state_dict()
    best_epoch = -1
    since = time.time()

    train_consistent_mean, train_consistent_std = get_consistent_mean(train_paths, train_labels)
    test_consistent_mean, test_consistent_std = get_consistent_mean(test_paths, test_labels)
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }

    train_data = TrainDataset(train_paths, train_labels, data_transforms['train'])
    test_data = TrainDataset(test_paths, test_labels, data_transforms['test'])
    img_datasets = {'train': train_data, 'test': test_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5,
                                    num_workers=1)
    test_data_loaders = DataLoader(test_data, shuffle=True, batch_size=5,
                                  num_workers=1)
    dataloader = {'train': train_data_loaders, 'test': test_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    for epoch in range(num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels) in enumerate(dataloader[phase]):
                traditional_feats = [img_feats_dict[path] for path in paths]
                traditional_feats = np.asarray(traditional_feats)
                traditional_feats = traditional_feats.astype(np.float64)
                traditional_feats = torch.Tensor(traditional_feats)

                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    traditional_feats = traditional_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward((inputs, traditional_feats))

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            if phase == 'train':
                train_loss = running_loss / len(img_datasets[phase])
                train_acc = running_corrects / len(img_datasets[phase])
                print('-'*25, '*', '-'*25)
                print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, train_loss, train_acc))
            else:
                pass

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")


def get_img_feats_dict(feats_csv_file):

    df = pd.read_csv(feats_csv_file)
    img_feats_dict = dict()
    for index, row in df.iterrows():
        img_name = row['image'].replace('csv', 'jpg')
        img_feats = np.asarray(row[3:], dtype=np.float64)
        img_feats_dict[img_name] = img_feats
    print('len of feats dict:', len(img_feats_dict))
    # print(img_feats_dict)
    return img_feats_dict


def train_pipeline(*data):
    num_epochs = 200

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])
    if use_gpu:
        weights = weights.cuda(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

    feats_csv_file = global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file)

    model = models.densenet121(pretrained=True)
    model = Net4(model, num_extra_feats=10)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if use_gpu:
        model = model.cuda(device)

    model_name = 'densenet121_w_trfm'
    train_model(model, *data, num_epochs, criterion, optimizer, scheduler, use_gpu,
            model_name=model_name, img_feats_dict=img_feats_dict)
    

if __name__ == '__main__':

    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=global_var.base_data_prefix + '/roi')
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=global_var.base_test_data_prefix + '/roi')
    print(len(test_img_labels))
     
    train_pipeline(train_img_paths, train_img_labels, test_img_paths, test_img_labels)

