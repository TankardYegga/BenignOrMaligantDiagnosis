# -*- encoding: utf-8 -*-
"""
@File    : train20.py
@Time    : 2/24/2022 10:47 AM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
交叉验证的ConvNeXt V2 结果
"""
from collections import OrderedDict

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

from convnext_v2 import convnext_tiny

try:
    from PIL import Image
except ImportError:
    print(ImportError)
    import Image


def get_k_fold_data(k, i, x, y):
    """
    :param k:
    :param i: k折数据中第i折
    :param x:
    :param y:
    :return:
    """
    assert k > 1
    fold_size = len(x) // k
    x_train, y_train = None, None
    x_val, y_val = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[idx], y[idx]
        if j == i:
            x_val, y_val = x_part, y_part
        elif x_train == None and y_train == None:
            x_train, y_train = x_part, y_part
        else:
            x_train = x_train + x_part
            y_train = y_train + y_part

    return x_train, y_train, x_val, y_val


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_img_data_from_path(img_path, loader=pil_loader):
    return loader(img_path)


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

        cur_img_name = os.path.basename(cur_img_path)[3:]
        return cur_img_name, cur_img_data, cur_img_label


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


def get_available_data_by_order(data_dir='../shuffled_merged_data2'):
    train_img_path = []
    train_img_label = []
    img_names_list = sorted(os.listdir(data_dir), key=cmp_to_key(cmp_img))
    for img_name in img_names_list:
        img_path = os.path.join(data_dir, img_name)
        label = 0 if img_name[3:].startswith('B') else 1
        train_img_path.append(img_path)
        train_img_label.append(label)
        # print(img_path)
        # print(label)

    # print(len(train_img_path))
    # print(train_img_label)
    # print(train_img_path)
    return train_img_path, train_img_label


def get_consistent_mean(img_paths, img_labels):
    img_paths = sorted(img_paths, key=cmp_to_key(cmp_img))
    img_labels = [1 if 'M' in img_path else 0 for img_path in img_paths]

    consistent_mean_sum = torch.zeros(3)
    consistent_std_sum = torch.zeros(3)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    for img_path in img_paths:
        correspond_img_name = os.path.basename(img_path)[3:]
        correspond_img_path = os.path.join('../merged_data/', correspond_img_name)
        img_data = pil_loader(correspond_img_path)
        img_data = data_transforms(img_data)
        # print(type(img_data))
        # print(img_data.size())
        for i in range(3):
            consistent_mean_sum[i] += img_data[i, :, :].mean()
            consistent_std_sum[i] += img_data[i, :, :].std()

    return consistent_mean_sum/len(img_paths), consistent_std_sum/len(img_paths)


def train_model(i, net, train_paths, train_labels, val_paths, val_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                model_name='', img_feats_dict=dict(), pkl_file='',
                ):
    """
    :param i:
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
    print('*' * 15, str(i) + ' fold', '*' * 15)
    global val_acc, val_loss, train_acc, train_loss
    start_epoch = 0
    print('pickle file is:', pkl_file)
    print(os.path.exists(pkl_file))
    if os.path.exists(pkl_file):
        net.load_state_dict(torch.load(pkl_file))
        # start_epoch = os.path.basename(pkl_file).split('_')[-1].split('.')[0]
        start_epoch = int(os.path.basename(pkl_file).split('_')[-2])
        print('the start epoch is:', start_epoch)
    else:
        print('the start epoch is:', start_epoch)
        # return

    model_pkls_name = 'pkls_' + model_name + '_' + str(num_epochs)
    print("model_pkls_name:", model_pkls_name)
    dir_path = model_pkls_name
    # dir_path = model_pkls_name
    print("dir:", dir_path)
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

    # consistent_mean = torch.Tensor([0.0411, 0.0411, 0.0411])
    # consistent_std = torch.Tensor([0.0195, 0.0195, 0.0195])
    train_consistent_mean, train_consistent_std = get_consistent_mean(train_paths, train_labels)
    val_consistent_mean, val_consistent_std = get_consistent_mean(val_paths, val_labels)
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_consistent_mean, val_consistent_std)
        ]),
    }

    train_data = TrainDataset(train_paths, train_labels, data_transforms['train'])
    val_data = TrainDataset(val_paths, val_labels, data_transforms['val'])
    img_datasets = {'train': train_data, 'val': val_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5,
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=5,
                                  num_workers=1)
    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}

    best_epoch = -1
    best_val_acc = -1
    best_train_acc = -1
    best_train_loss = -1
    best_val_loss = -1
    best_model_wts = net.state_dict()
    for epoch in range(num_epochs):

        running_stages = ['train', 'val']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels) in enumerate(dataloader[phase]):

                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                outputs = net.forward(inputs)
                # print(outputs)
                # print(type(outputs))
                # print(outputs.shape)
                # print(dir(outputs))
                # print(outputs.data)
                _, preds = torch.max(outputs.data, 1)

                # preds_one_hot = torch.eye(2)[preds]
                # preds_one_hot = Variable(preds_one_hot, requires_grad=True)
                # labels_one_hot = torch.eye(2)[labels]
                # loss = criterion(preds_one_hot, labels_one_hot)
                # print('loss is', loss)
                # print(type(loss))
                # print(type(loss))
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                # print(loss)
                # print(type(loss))
                # print(loss.shape)
                # print(loss.data == loss)
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            if phase == 'train':
                train_loss = running_loss / len(img_datasets[phase])
                train_acc = running_corrects / len(img_datasets[phase])
                print('-'*25, '*', '-'*25)
                print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, train_loss, train_acc))
            else:
                val_loss = running_loss / len(img_datasets[phase])
                val_acc = running_corrects / len(img_datasets[phase])
                print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, val_loss, val_acc))
                print('-'*25, '*', '-'*25)

        if val_acc > best_val_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_acc = train_acc
            best_train_loss = train_loss

    save_path = model_pkls_name + '/' + str(i) + '_' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_val_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    return best_val_acc, best_val_loss, best_train_acc, best_train_loss


def get_img_feats_dict(feats_csv_file):

    df = pd.read_csv(feats_csv_file)
    img_feats_dict = dict()
    for index, row in df.iterrows():
        img_name = row['image'].replace('csv', 'bmp')
        img_feats = np.asarray(row[3:])
        img_feats_dict[img_name] = img_feats
    print('len of feats dict:', len(img_feats_dict))
    print(img_feats_dict)
    return img_feats_dict


def k_fold_train(k, train_img_path, train_img_label):
    num_epochs = 200

    use_gpu = torch.cuda.is_available()
    weights = torch.FloatTensor([0.4, 0.6])
    if use_gpu:
        weights = weights.cuda()
        criterion = nn.CrossEntropyLoss(weight=weights)

    feats_csv_file = '../merged_features/filtered_features.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file)

    train_loss_sum = 0.0
    train_acc_sum = 0.0
    val_loss_sum = 0.0
    val_acc_sum = 0.0

    best_acc = 0.0
    best_epoch_th = -1

    for i in range(3, k):
        ith_fold_data = get_k_fold_data(k, i, train_img_path, train_img_label)

        model = convnext_tiny(2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        if use_gpu:
            model = model.cuda()

        model_name = 'covnext_v2_cv_' + str(k)
        best_val_acc, best_val_loss, best_train_acc, best_train_loss = \
            train_model(i, model, *ith_fold_data, num_epochs, criterion, optimizer, scheduler, use_gpu,
                model_name=model_name, img_feats_dict=img_feats_dict)

        train_loss_sum += best_train_loss
        train_acc_sum += best_train_acc
        val_loss_sum += best_val_loss
        val_acc_sum += best_val_acc

        if best_val_acc > best_acc:
            best_acc = best_val_acc
            best_epoch_th = i

        model_pkls_name = 'pkls_' + model_name + '_' + str(num_epochs)
        with open(os.path.join(model_pkls_name, 'cross_validation_results.txt'), "a") as f:
            f.write(str(i) + ' fold\n')
            f.write('train loss:%.4f' % best_train_loss + 'train_acc:%.4f' % best_train_acc + "\n")
            f.write('test loss:%.4f' % best_val_loss + 'val acc:%.4f' % best_val_acc + "\n")

    print('*'*10, '最终交叉验证结果', '*'*10)
    print('best fold:', best_epoch_th)
    print('best acc:', best_acc)
    print('train_loss:%.2f' % (train_loss_sum/k), 'train_acc:%.2f' % (train_acc_sum/k))
    print('val_loss:%.2f' % (val_loss_sum/k), 'val_acc:%.2f' % (val_acc_sum/k))
    print('*'*10, '最终交叉验证结果', '*'*10)


if __name__ == '__main__':
    train_img_paths, train_img_labels = get_available_data_by_order()
    k = 10
    k_fold_train(k, train_img_paths, train_img_labels)