# 直接在原来的整个数据集上训练
# 然后在测试集上进行测试
from collections import OrderedDict
import imp
from math import inf
from operator import mod
import random
from regex import P

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

import shutil
from train_and_test_copy_7 import setup_seed
from test_final import DatasetWithSingleImage, TestDatasetWithThresholdingData

from torch.utils.data import DataLoader, Dataset

from torch.optim import lr_scheduler
from classfication_metrics import cal_metrics
from test_final import DatasetWithAddedData, DatasetWithMaskData, DatasetWithMaskDataOffline, DatasetWithMaskDataOnly, DatasetWithStackedData, TestDataset, TestDatasetWithMaskData, get_img_feats_dict

from densenet_plus import *
from test import extra_feats_pipeline
from __init__ import global_var
import pickle

try:
    from PIL import Image
except ImportError:
    print(ImportError)
    import Image

device = "cuda:5"

# device = ""

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

    # img_names_list = os.listdir(data_dir)

    img_names_list = sorted(os.listdir(data_dir), key=lambda x: get_number(x))

    # print('names:', img_names_list)

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
    print("finished")

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


def train_model(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
               img_feats_dict=dict(), model_name='',  pkl_file='',
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

    suffix = list(img_feats_dict.keys())[0][-3:]

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
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = TrainDataset(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

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
                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)

                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward((inputs, extra_feats))

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
                str(model_name) + '_epoch_' + str(epoch) + '.pkl'
            torch.save(best_model_wts, save_path)

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path


def train_model_w_feats_and_topo_mask(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                model_name='', pkl_file='',
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
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = TestDatasetWithMaskData(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

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

            for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(dataloader[phase]):

                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward((inputs, extra_feats, topo_mask_data))

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
                str(model_name) + '_epoch_' + str(epoch) + '.pkl'
            torch.save(best_model_wts, save_path)

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path



def train_model_w_topo_mask_data(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='',
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

    suffix = list(img_feats_dict.keys())[0][-3:]

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
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = DatasetWithMaskData(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

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

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward((inputs, extra_feats, topo_mask_data))

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
                str(model_name) + '_epoch_' + str(epoch) + '.pkl'
            torch.save(best_model_wts, save_path)

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path



def train_model_w_topo_mask_offline0(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()


    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward((inputs, extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
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

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(best_model_wts, save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
        

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path



def train_model_w_topo_mask_offline(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file), strict=False)
        
        print("parameters load")

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])


    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }


    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}


    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=2,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}


    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()


    count = 0


    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
             
                if len(paths) == 1 and skipped_one:
                    continue
                outputs = net.forward((inputs, extra_feats, topo_mask_data))
    
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                for predth in range(preds.shape[0]):
                    preds_list.append(preds[predth].item())
                    labels_list.append(labels[predth].item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                running_loss += loss.data
                # running_corrects += torch.sum(preds == labels.data).to(torch.float32)
                # running_uncorrects += torch.sum(preds != labels.data).to(torch.float32)


            correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
            uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()

            print('-'*25, '*', '-'*25)

            print("correct num:", correct_num)
            print("uncorrect num:", uncorrect_num)
            print('train datasets length:', (correct_num + uncorrect_num))
            print('train acc:', correct_num / (correct_num + uncorrect_num))
            
            assert (correct_num + uncorrect_num) == len(labels_list)
            assert len(preds_list) == len(labels_list)

            labels_arr = np.asarray(labels_list)
            preds_arr = np.asarray(preds_list)
            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': preds_arr}
            dict_filename = str(epoch) + '_train_dicts.pickle'
            save_dir = dir_path + '/saved_dicts/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_path = os.path.join(save_dir, dict_filename)
            with open(save_path, 'wb') as fp:
                pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

            if phase == 'train':

                train_loss = running_loss / (len(preds_list) )
                train_acc = correct_num / (len(preds_list) )

                # print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, train_loss, train_acc))

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25, '*', '-'*25)

            else:
                pass

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '.pkl'
        torch.save(net.state_dict(), save_path)

        if train_acc >= best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 10:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path



def train_model_w_topo_mask_offline2(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        print('parameter loaded!')

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    test_consistent_mean = train_consistent_mean
    test_consistent_std = train_consistent_std

    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

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

    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    test_data = train_data

    img_datasets = {'train': train_data, 'test': test_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    test_data_loaders = DataLoader(test_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders, 'test': test_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0
            running_corrects = 0.0
            running_uncorrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
                
                # print("paths:", paths)
                # print("topo_mask_data:", topo_mask_data.shape)
                if len(paths) == 1 and skipped_one:
                    continue
                outputs = net.forward((inputs, extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                for predth in range(preds.shape[0]):
                    preds_list.append(preds[predth].item())
                    labels_list.append(labels[predth].item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)
                running_uncorrects += torch.sum(preds != labels.data).to(torch.float32)

            # labels_arr = np.asarray(labels_list)
            # preds_arr = np.asarray(preds_list)
            # auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            # dict = {'labels': labels_arr, 'preds': preds_arr}
            # dict_filename = str(epoch) + '_train_dicts.pickle'
            # save_dir = dir_path + '/saved_dicts/'
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)

            # save_path = os.path.join(save_dir, dict_filename)
            # with open(save_path, 'wb') as fp:
            #     pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

            # print("running corrects:", running_corrects)
            # print("running uncorrects:", running_uncorrects)
            # print("preds:", preds_list)
            # print("len preds:", len(preds_list))
            # print("acc2", sum(np.array(preds_list) == np.array(labels_list)))

            if phase == 'train':
               
                # train_loss = running_loss / (len(preds_list) )
                # train_acc = running_corrects / (len(preds_list) )

                train_loss = running_loss / len(img_datasets[phase])
                train_acc = running_corrects / len(img_datasets[phase])

                print('-'*25, '*', '-'*25)
                print('{} Loss:{:.4f} Acc:{:.4f} '.format(phase, train_loss, train_acc))

                # print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, auc_score))
                # print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                # print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))
            else:
                # val_loss = running_loss / (len(preds_list) )
                # val_acc = running_corrects / (len(preds_list) )
                val_loss = running_loss / len(img_datasets[phase])
                val_acc = running_corrects / len(img_datasets[phase])
                print('-'*25, '*', '-'*25)
                print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, val_loss, val_acc))

                # print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, auc_score))
                # print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                # print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))
                pass

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(net.state_dict(), save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 5:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path


def train_model_w_topo_mask_offline3(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        print('parameter loaded!')

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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

    test_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/test_part_of_original_data2/roi'
    test_img_paths, test_img_labels = get_available_data_by_order(data_dir=test_dir)

    train_consistent_mean, train_consistent_std = get_consistent_mean(train_paths, train_labels)
    # test_consistent_mean, test_consistent_std = get_consistent_mean(test_img_paths, test_img_labels)
    test_consistent_mean = train_consistent_mean
    test_consistent_std = train_consistent_std

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

    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    # test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    test_data =  TestDatasetWithMaskData(train_paths, train_labels, data_transforms['test'])

    img_datasets = {'train': train_data, 'test': test_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    test_data_loaders = DataLoader(test_data, shuffle=True, batch_size=1,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders, 'test': test_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0
            running_corrects = 0.0
            running_uncorrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
                
                # print("paths:", paths)
                # print("topo_mask_data:", topo_mask_data.shape)
                if len(paths) == 1 and skipped_one:
                    continue
                outputs = net.forward((inputs, extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                for predth in range(preds.shape[0]):
                    preds_list.append(preds[predth].item())
                    labels_list.append(labels[predth].item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                running_loss += loss.data
                # running_corrects += torch.sum(preds == labels.data).to(torch.float32)
                # running_uncorrects += torch.sum(preds != labels.data).to(torch.float32)

            # labels_arr = np.asarray(labels_list)
            # preds_arr = np.asarray(preds_list)
            # auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            # dict = {'labels': labels_arr, 'preds': preds_arr}
            # dict_filename = str(epoch) + '_train_dicts.pickle'
            # save_dir = dir_path + '/saved_dicts/'
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)

            # save_path = os.path.join(save_dir, dict_filename)
            # with open(save_path, 'wb') as fp:
            #     pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

            # print("running corrects:", running_corrects)
            # print("running uncorrects:", running_uncorrects)
            # print("preds:", preds_list)
            # print("len preds:", len(preds_list))
            # print("acc2", sum(np.array(preds_list) == np.array(labels_list)))

            running_corrects = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
            running_uncorrects = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()

            print("correct num:", running_corrects)
            print("uncorrect num:", running_uncorrects)
            print('train datasets length:', (running_uncorrects + running_corrects))
            print("len of arr:", len(labels_list))

            if phase == 'train':
               
                train_loss = running_loss / (running_uncorrects + running_corrects)
                train_acc = running_corrects / (running_uncorrects + running_corrects)

                print('-'*25, '*', '-'*25)
                print('{} Loss:{:.4f} Acc:{:.4f} '.format(phase, train_loss, train_acc))

                # print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, auc_score))
                # print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                # print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))
        
        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '.pkl'
        torch.save(net.state_dict(), save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 5:
                print("Training Process ends ")
                break
        
        preds_test_list = []
        labels_test_list = []
        paths_test_list = []

        net.train(True)
        
        for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_data_loaders):

            if use_gpu:
                # print('yes gpu used')
                inputs = inputs.cuda(device)
                labels = labels.cuda(device)
                extra_feats = extra_feats.cuda(device)
                topo_mask_data = topo_mask_data.cuda(device)
                net = net.cuda(device)

            outputs = net.forward((inputs, extra_feats, topo_mask_data))

            _, preds = torch.max(outputs.data, 1)
            preds_test_list.append(preds.data[0].item())
            print("the preds is:", preds.data)
            loss = criterion(outputs, labels)
            print("the labels is:", labels.data)
            labels_test_list.append(labels.data[0].item())
            paths_test_list.append(paths)
            print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            # if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            #     benign_mis_classified.append(paths)
            # if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            #     malignant_mis_classified.append(paths)

        correct_num = torch.sum(torch.Tensor(labels_test_list) == torch.Tensor(preds_test_list)).item()
        uncorrect_num = torch.sum(torch.Tensor(labels_test_list) != torch.Tensor(preds_test_list)).item()
        print("correct num:", correct_num)
        print("uncorrect num:", uncorrect_num)
        print('train datasets length:', (correct_num + uncorrect_num))
        
        test_loss = running_loss / (correct_num + uncorrect_num)
        test_acc = correct_num / (correct_num + uncorrect_num)
        
        print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))
               

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path




def train_model_w_val_results(net, train_paths, train_labels, val_paths, val_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    val_consistent_mean, val_consistent_std = get_consistent_mean(val_paths, val_labels)

    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'val':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_consistent_mean, val_consistent_std)
        ])
    }

    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    val_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['val'])

    img_datasets = {'train': train_data, 'val': val_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=10, 
                                    num_workers=1)
    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train', 'val']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
                
                # print("paths:", paths)
                # print("topo_mask_data:", topo_mask_data.shape)
                if len(paths) == 1 and skipped_one:
                    continue
                outputs = net.forward((inputs, extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
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
                val_loss = running_loss / len(img_datasets[phase])
                val_acc = running_corrects / (len(img_datasets[phase]))
                print('-'*25, '*', '-'*25)
                print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, train_loss, train_acc))

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(net.state_dict(), save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 10:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path



def train_model_w_topo_mask_offline_2(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward((extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
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

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(best_model_wts, save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 5:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path



def train_model_w_topo_mask_offline_3(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = DatasetWithStackedData(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward((inputs, extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
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

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(best_model_wts, save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 5:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path


def train_model_w_topo_mask_offline_4(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = DatasetWithAddedData(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward((inputs, extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
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

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(best_model_wts, save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 5:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path




def train_topo_model(net, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                 model_name='',  pkl_file='', epoch_pos=-1
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

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = DatasetWithMaskDataOnly(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, labels, topo_mask_data) in enumerate(dataloader[phase]):

                if use_gpu:
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs = net.forward(topo_mask_data)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
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

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(best_model_wts, save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 5:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path



def train_model_with_topo_model(net, topo_model, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1, 
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)

                topo_mask_data = topo_mask_data.cuda(device)

                topo_mask_data = topo_model.conv1(topo_mask_data)
                topo_mask_data = topo_model.bn1(topo_mask_data)
                topo_mask_data = topo_model.relu(topo_mask_data)
                topo_mask_data = topo_model.maxpool(topo_mask_data)
                topo_mask_data = topo_model.layer1(topo_mask_data)
                topo_mask_data = topo_model.layer2(topo_mask_data)
                topo_mask_data = topo_model.layer3(topo_mask_data)
                topo_mask_data = topo_model.layer4(topo_mask_data)
                topo_mask_data = topo_model.avgpool(topo_mask_data).reshape(len(paths), 2048)
                topo_mask_data = topo_model.fc[0](topo_mask_data)
                topo_mask_data = topo_model.fc[1](topo_mask_data)

                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
               
                outputs = net.forward((inputs, extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
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

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(best_model_wts, save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 5:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path


def train_model_with_topo_model2(net, topo_model, train_paths, train_labels,
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                img_feats_dict, model_name='',  pkl_file='', epoch_pos=-1, 
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

    suffix = list(img_feats_dict.keys())[0][-3:]

    start_epoch = 0

    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file))

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    img_datasets = {'train': train_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=10,
                                    num_workers=1)
    dataloader = {'train': train_data_loaders}

    best_epoch = -1
    best_train_acc = -1
    best_train_loss = -1
    best_model_wts = net.state_dict()

    count = 0

    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                extra_feats = [img_feats_dict[path] if suffix in path else img_feats_dict[path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)

                topo_mask_data = topo_mask_data.cuda(device)

                topo_mask_data = topo_model.conv1(topo_mask_data)
                topo_mask_data = topo_model.bn1(topo_mask_data)
                topo_mask_data = topo_model.relu(topo_mask_data)
                topo_mask_data = topo_model.maxpool(topo_mask_data)
                topo_mask_data = topo_model.layer1(topo_mask_data)
                topo_mask_data = topo_model.layer2(topo_mask_data)
                topo_mask_data = topo_model.layer3(topo_mask_data)
                topo_mask_data = topo_model.layer4(topo_mask_data)
                topo_mask_data = topo_model.avgpool(topo_mask_data).reshape(len(paths), 2048)
                topo_mask_data = topo_model.fc[0](topo_mask_data)

                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
               
                outputs = net.forward((inputs, extra_feats, topo_mask_data))
                # print("outputs is:", outputs)
                # print("labels is:", labels)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
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

        save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc.item()) + '.pkl'
        torch.save(best_model_wts, save_path)

        if train_acc > best_train_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_train_acc = train_acc
            best_train_loss = train_loss
            count = 0
        else:
            count +=1
            if count >= 5:
                print("Training Process ends ")
                break

    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc.item()) + '_best.pkl'
    torch.save(best_model_wts, save_path)

    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Train Loss:", best_train_loss)
    print("Train Acc:", best_train_acc)
    print("---------------------train end----------------------")
    return save_path



####################################################################################################################
#
#
#  对训练和测试代码的重写：加入了验证集的部分
#
#
#
####################################################################################################################

import matplotlib.pyplot as plt


def draw_fig(list,name,epoch, save_path):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.title('loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.grid()
        plt.savefig(save_path)
        plt.show()
    elif name =="acc":
        plt.cla()
        plt.title('accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        plt.grid()
        plt.savefig(save_path)
        plt.show()



def worker_init_fn(worked_id):
    # print("initial seed", torch.initial_seed())
    # worker_seed = torch.initial_seed() % 2**32
    worker_seed = 10
    print("wrok seed:", worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_model_w_validation_part(
                net, train_paths, train_labels,
                val_paths, val_labels, 
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                train_img_feats_dict, val_img_feats_dict,
                model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False
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
    global val_acc, val_loss, train_acc, train_loss

    seed = 10
    setup_seed(seed)

    suffix = list(train_img_feats_dict.keys())[0][-3:]

    start_epoch = 0
    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file), strict=False)
        
        print("parameters load")

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    else:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.mkdir(dir_path)


    best_acc = 0.0
    best_model_wts = net.state_dict()
    best_epoch = -1
    since = time.time()

    train_consistent_mean, train_consistent_std = get_consistent_mean(train_paths, train_labels)
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])
    val_consistent_mean, val_consistent_std = get_consistent_mean(val_paths, val_labels)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_consistent_mean, val_consistent_std),
        ]),
    }


    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    val_data = DatasetWithMaskDataOffline(val_paths, val_labels, data_transforms['val'])

    img_datasets = {'train': train_data, 'val': val_data}

    # g=torch.Generator()
    # g.manual_seed(10)
    # train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                                 num_workers=1)
    # g.manual_seed(10)
    # val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                         num_workers=1)

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5, 
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=5,  
                            num_workers=1)

    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}


    best_epoch = -1
    best_val_acc = -1
    best_train_acc = -1
    best_val_loss = float(inf)
    best_model_wts = net.state_dict()
    best_save_path = ''
    
    feats_dict = {'train': train_img_feats_dict, 'val': val_img_feats_dict}

    count = 0


    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    
    for epoch in range(start_epoch, num_epochs):
    # for epoch in range(start_epoch, 10):

        running_stages = ['train', 'val']
        # running_stages = ['train']

        print('**'*15, str(epoch) + "starting ", '**' * 15)

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                # if j == 0:
                #     print("paths: ", paths)

                extra_feats = [feats_dict[phase][path] if suffix in path else feats_dict[phase][path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
           
                outputs = net.forward((inputs, extra_feats, topo_mask_data))
    
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                for predth in range(preds.shape[0]):
                    preds_list.append(preds[predth].item())
                    labels_list.append(labels[predth].item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    pass

                running_loss += loss.data
            
            correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
            uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()

            print('-'*25, '*', '-'*25)

            # print("correct num:", correct_num)
            # print("uncorrect num:", uncorrect_num)
            # print('train datasets length:', (correct_num + uncorrect_num))
            # print('train acc:', correct_num / (correct_num + uncorrect_num))
            
            assert (correct_num + uncorrect_num) == len(labels_list)
            assert len(preds_list) == len(labels_list)

            labels_arr = np.asarray(labels_list)
            preds_arr = np.asarray(preds_list)
            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': preds_arr}
            dict_filename = str(epoch) + '_' + str(phase) + '_dicts.pickle'
            save_dir = dir_path + '/saved_dicts/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)


            save_path = os.path.join(save_dir, dict_filename)
            with open(save_path, 'wb') as fp:
                pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


            if phase == 'train':

                print('-'*25)

                train_loss = running_loss / (len(preds_list) )
                train_acc = correct_num / (len(preds_list) )

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25, '*', '-'*25)


        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss.item())
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss.item())


        # if val_loss <= best_val_loss:
        if val_acc >= best_val_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_acc = train_acc

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)
            best_save_path = save_path

            # save_path = dir_path + '/' + \
            # str(model_name) + '_epoch2_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            # torch.save(best_model_wts, save_path)
        else:
            count += 1
        #     if count >= 10:
        #         print("Training Process ends ")
        #         break


    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_' + str(best_val_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)

#################################################################################


    loss_and_acc_saved_dir = os.path.join(dir_path, 'lossAndacc')
    if not os.path.exists(loss_and_acc_saved_dir):
        os.mkdir(loss_and_acc_saved_dir)
    else:
        print(loss_and_acc_saved_dir, 'already exists!')


    train_acc_saved_path = os.path.join(loss_and_acc_saved_dir, model_name + '_train_acc.png')
    train_loss_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_train_loss.png')
    # draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    # draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)


    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    # draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    # draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("---------------------train end----------------------")
    # return save_path
    return best_save_path



def train_model_w_validation_part2(
                net, train_paths, train_labels,
                val_paths, val_labels, 
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                train_img_feats_dict, val_img_feats_dict,
                model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False, w_loss = 1.1
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
    global val_acc, val_loss, train_acc, train_loss

    seed = 10
    setup_seed(seed)

    suffix = list(train_img_feats_dict.keys())[0][-3:]

    start_epoch = 0
    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file), strict=False)
        
        print("parameters load")

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    else:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.mkdir(dir_path)


    best_acc = 0.0
    best_model_wts = net.state_dict()
    best_epoch = -1
    since = time.time()

    train_consistent_mean, train_consistent_std = get_consistent_mean(train_paths, train_labels)
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])
    val_consistent_mean, val_consistent_std = get_consistent_mean(val_paths, val_labels)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_consistent_mean, val_consistent_std),
        ]),
    }


    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    val_data = DatasetWithMaskDataOffline(val_paths, val_labels, data_transforms['val'])

    img_datasets = {'train': train_data, 'val': val_data}

    # g=torch.Generator()
    # g.manual_seed(10)
    # train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                                 num_workers=1)
    # g.manual_seed(10)
    # val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                         num_workers=1)

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5, 
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=5,  
                            num_workers=1)

    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}


    best_epoch = -1
    best_val_acc = -1
    best_train_acc = -1
    best_val_loss = float(inf)
    best_model_wts = net.state_dict()
    best_save_path = ''
    
    feats_dict = {'train': train_img_feats_dict, 'val': val_img_feats_dict}

    count = 0


    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    
    for epoch in range(start_epoch, num_epochs):
    # for epoch in range(start_epoch, 10):

        running_stages = ['train', 'val']
        # running_stages = ['train']

        print('**'*15, str(epoch) + "starting ", '**' * 15)

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                # if j == 0:
                #     print("paths: ", paths)

                # extra_feats = [feats_dict[phase][path] if suffix in path else feats_dict[phase][path.replace(path[-3:], suffix)] for path in paths]
                # extra_feats = np.asarray(extra_feats)
                # extra_feats = extra_feats.astype(np.float64)
                # extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    # topo_mask_data = topo_mask_data.cuda(device)
                    # extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()


                # outputs, outputs_1, outputs_2 = net.forward((inputs, extra_feats, topo_mask_data))
                outputs, outputs_1, outputs_2 = net.forward((inputs, paths))


                _, preds = torch.max(outputs.data, 1)


                loss = w_loss * criterion(outputs, labels) + criterion(outputs_1, labels) +  criterion(outputs_2, labels)

        
                loss = criterion(outputs, labels)

                for predth in range(preds.shape[0]):
                    preds_list.append(preds[predth].item())
                    labels_list.append(labels[predth].item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    pass

                running_loss += loss.data
            
            correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
            uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()

            print('-'*25, '*', '-'*25)

            # print("correct num:", correct_num)
            # print("uncorrect num:", uncorrect_num)
            # print('train datasets length:', (correct_num + uncorrect_num))
            # print('train acc:', correct_num / (correct_num + uncorrect_num))
            
            assert (correct_num + uncorrect_num) == len(labels_list)
            assert len(preds_list) == len(labels_list)

            labels_arr = np.asarray(labels_list)
            preds_arr = np.asarray(preds_list)
            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': preds_arr}
            dict_filename = str(epoch) + '_' + str(phase) + '_dicts.pickle'
            save_dir = dir_path + '/saved_dicts/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)


            save_path = os.path.join(save_dir, dict_filename)
            with open(save_path, 'wb') as fp:
                pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


            if phase == 'train':

                print('-'*25)

                train_loss = running_loss / (len(preds_list) )
                train_acc = correct_num / (len(preds_list) )

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25, '*', '-'*25)


        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss.item())
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss.item())


        # if val_loss <= best_val_loss:
        if val_acc >= best_val_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_acc = train_acc

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)
            best_save_path = save_path

            # print(net.res_weights_dict)
            # print(net.dense_weights_dict)

            # save_path = dir_path + '/' + \
            # str(model_name) + '_epoch2_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            # torch.save(best_model_wts, save_path)
        else:
            count += 1
        #     if count >= 10:
        #         print("Training Process ends ")
        #         break


    ###  这段代码是错误的，不需要，因为其始终保存的是最后一次迭代的模型检查点
    # save_path = dir_path + '/' + \
    #             str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_' + str(best_val_acc) + '_best.pkl'
    # torch.save(best_model_wts, save_path)

#################################################################################


    loss_and_acc_saved_dir = os.path.join(dir_path, 'lossAndacc')
    if not os.path.exists(loss_and_acc_saved_dir):
        os.mkdir(loss_and_acc_saved_dir)
    else:
        print(loss_and_acc_saved_dir, 'already exists!')


    train_acc_saved_path = os.path.join(loss_and_acc_saved_dir, model_name + '_train_acc.png')
    train_loss_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_train_loss.png')
    # draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    # draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)


    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    # draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    # draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("---------------------train end----------------------")
    # return save_path
    return best_save_path




def train_model_w_validation_part22(
                net, train_paths, train_labels,
                val_paths, val_labels, 
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                train_img_feats_dict, val_img_feats_dict,
                model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False, w_loss = 1.1
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
    global val_acc, val_loss, train_acc, train_loss

    seed = 10
    setup_seed(seed)

    suffix = list(train_img_feats_dict.keys())[0][-3:]

    start_epoch = 0
    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file), strict=False)
        
        print("parameters load")

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    else:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.mkdir(dir_path)


    best_acc = 0.0
    best_model_wts = net.state_dict()
    best_epoch = -1
    since = time.time()

    train_consistent_mean, train_consistent_std = get_consistent_mean(train_paths, train_labels)
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])
    val_consistent_mean, val_consistent_std = get_consistent_mean(val_paths, val_labels)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_consistent_mean, val_consistent_std),
        ]),
    }


    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    val_data = DatasetWithMaskDataOffline(val_paths, val_labels, data_transforms['val'])

    img_datasets = {'train': train_data, 'val': val_data}

    # g=torch.Generator()
    # g.manual_seed(10)
    # train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                                 num_workers=1)
    # g.manual_seed(10)
    # val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                         num_workers=1)

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=3, 
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=3,  
                            num_workers=1)

    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}


    best_epoch = -1
    best_val_acc = -1
    best_train_acc = -1
    best_val_loss = float(inf)
    best_model_wts = net.state_dict()
    best_save_path = ''
    
    feats_dict = {'train': train_img_feats_dict, 'val': val_img_feats_dict}

    count = 0


    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    
    for epoch in range(start_epoch, num_epochs):
    # for epoch in range(start_epoch, 10):

        running_stages = ['train', 'val']
        # running_stages = ['train']

        print('**'*15, str(epoch) + "starting ", '**' * 15)

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                # if j == 0:
                #     print("paths: ", paths)

                # extra_feats = [feats_dict[phase][path] if suffix in path else feats_dict[phase][path.replace(path[-3:], suffix)] for path in paths]
                # extra_feats = np.asarray(extra_feats)
                # extra_feats = extra_feats.astype(np.float64)
                # extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    # topo_mask_data = topo_mask_data.cuda(device)
                    # extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()


                # outputs, outputs_1, outputs_2 = net.forward((inputs, extra_feats, topo_mask_data))
                outputs, outputs_1, outputs_2, weights = net.forward((inputs, paths))


                _, preds = torch.max(outputs.data, 1)


                loss = w_loss * criterion(outputs, labels) + criterion(outputs_1, labels) +  criterion(outputs_2, labels)

        
                loss = criterion(outputs, labels)

                for predth in range(preds.shape[0]):
                    preds_list.append(preds[predth].item())
                    labels_list.append(labels[predth].item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    pass

                running_loss += loss.data
            
            correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
            uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()

            print('-'*25, '*', '-'*25)

            # print("correct num:", correct_num)
            # print("uncorrect num:", uncorrect_num)
            # print('train datasets length:', (correct_num + uncorrect_num))
            # print('train acc:', correct_num / (correct_num + uncorrect_num))
            
            assert (correct_num + uncorrect_num) == len(labels_list)
            assert len(preds_list) == len(labels_list)

            labels_arr = np.asarray(labels_list)
            preds_arr = np.asarray(preds_list)
            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': preds_arr}
            dict_filename = str(epoch) + '_' + str(phase) + '_dicts.pickle'
            save_dir = dir_path + '/saved_dicts/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)


            save_path = os.path.join(save_dir, dict_filename)
            with open(save_path, 'wb') as fp:
                pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


            if phase == 'train':

                print('-'*25)

                train_loss = running_loss / (len(preds_list) )
                train_acc = correct_num / (len(preds_list) )

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25, '*', '-'*25)


        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss.item())
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss.item())


        # if val_loss <= best_val_loss:
        if val_acc >= best_val_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_acc = train_acc

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)
            best_save_path = save_path

            # save_path = dir_path + '/' + \
            # str(model_name) + '_epoch2_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            # torch.save(best_model_wts, save_path)
        else:
            count += 1
        #     if count >= 10:
        #         print("Training Process ends ")
        #         break

    loss_and_acc_saved_dir = os.path.join(dir_path, 'lossAndacc')
    if not os.path.exists(loss_and_acc_saved_dir):
        os.mkdir(loss_and_acc_saved_dir)
    else:
        print(loss_and_acc_saved_dir, 'already exists!')


    train_acc_saved_path = os.path.join(loss_and_acc_saved_dir, model_name + '_train_acc.png')
    train_loss_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_train_loss.png')
    # draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    # draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)


    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    # draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    # draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("---------------------train end----------------------")
    return best_save_path




def train_model_w_validation_part21(
                net, train_paths, train_labels,
                val_paths, val_labels, 
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                train_img_feats_dict, val_img_feats_dict,
                model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False, w_loss = 1.1
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
    global val_acc, val_loss, train_acc, train_loss

    seed = 10
    setup_seed(seed)

    suffix = list(train_img_feats_dict.keys())[0][-3:]

    start_epoch = 0
    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file), strict=False)
        
        print("parameters load")

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    else:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.mkdir(dir_path)


    best_acc = 0.0
    best_model_wts = net.state_dict()
    best_epoch = -1
    since = time.time()

    train_consistent_mean, train_consistent_std = get_consistent_mean(train_paths, train_labels)
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])
    val_consistent_mean, val_consistent_std = get_consistent_mean(val_paths, val_labels)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_consistent_mean, val_consistent_std),
        ]),
    }


    train_data = DatasetWithMaskDataOffline(train_paths, train_labels, data_transforms['train'])
    val_data = DatasetWithMaskDataOffline(val_paths, val_labels, data_transforms['val'])

    img_datasets = {'train': train_data, 'val': val_data}

    # g=torch.Generator()
    # g.manual_seed(10)
    # train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                                 num_workers=1)
    # g.manual_seed(10)
    # val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                         num_workers=1)

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=2, 
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=2,  
                            num_workers=1)

    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}


    best_epoch = -1
    best_val_acc = -1
    best_train_acc = -1
    best_val_loss = float(inf)
    best_model_wts = net.state_dict()
    best_save_path = ''
    
    feats_dict = {'train': train_img_feats_dict, 'val': val_img_feats_dict}

    count = 0


    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

   
    # best_val_auc = inf
    # best_val_sensitivty = inf
    # best_val_specificity = inf
    # best_val_ppv = inf
    # best_val_npv = inf

    # best_train_auc = inf
    # best_train_sensitivty = inf
    # best_train_specificity = inf
    # best_train_ppv = inf
    # best_train_npv = inf
    
    
    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train', 'val']
        # running_stages = ['train']

        print('**'*15, str(epoch) + "starting ", '**' * 15)

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

          
                extra_feats = [feats_dict[phase][path] if suffix in path else feats_dict[phase][path.replace(path[-3:], suffix)] for path in paths]
                extra_feats = np.asarray(extra_feats)
                extra_feats = extra_feats.astype(np.float64)
                extra_feats = torch.Tensor(extra_feats)
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
             

                outputs, outputs_1, outputs_2 = net.forward((inputs, extra_feats, topo_mask_data))
    
                _, preds = torch.max(outputs.data, 1)


                loss = w_loss * criterion(outputs, labels) + criterion(outputs_1, labels) +  criterion(outputs_2, labels)


                for predth in range(preds.shape[0]):
                    preds_list.append(preds[predth].item())
                    labels_list.append(labels[predth].item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                running_loss += loss.data
            
            correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
            uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()

            print('-'*25, '*', '-'*25)

            # print("correct num:", correct_num)
            # print("uncorrect num:", uncorrect_num)
            # print('train datasets length:', (correct_num + uncorrect_num))
            # print('train acc:', correct_num / (correct_num + uncorrect_num))
            
            assert (correct_num + uncorrect_num) == len(labels_list)
            assert len(preds_list) == len(labels_list)

            labels_arr = np.asarray(labels_list)
            preds_arr = np.asarray(preds_list)
            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': preds_arr}
            dict_filename = str(epoch) + '_' + str(phase) + '_dicts.pickle'
            save_dir = dir_path + '/saved_dicts/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)


            save_path = os.path.join(save_dir, dict_filename)
            with open(save_path, 'wb') as fp:
                pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


            if phase == 'train':

                print('-'*25)

                train_loss = running_loss / (len(preds_list) )
                train_acc = correct_num / (len(preds_list) )

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, auc_score))

                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25, '*', '-'*25)


        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss.item())
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss.item())
        

        # if val_loss <= best_val_loss:
        if val_acc >= best_val_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_acc = train_acc

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)
            best_save_path = save_path


            # save_path = dir_path + '/' + \
            # str(model_name) + '_epoch2_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            # torch.save(best_model_wts, save_path)
        else:
            count += 1
        #     if count >= 10:
        #         print("Training Process ends ")
        #         break


    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_' + str(best_val_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)


    loss_and_acc_saved_dir = os.path.join(dir_path, 'lossAndacc')
    if not os.path.exists(loss_and_acc_saved_dir):
        os.mkdir(loss_and_acc_saved_dir)
    else:
        print(loss_and_acc_saved_dir, 'already exists!')


    train_acc_saved_path = os.path.join(loss_and_acc_saved_dir, model_name + '_train_acc.png')
    train_loss_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_train_loss.png')
    # draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    # draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)


    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    # draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    # draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("---------------------train end----------------------")
    # return save_path
    return best_save_path



def train_model_w_validation_part3(
                net, train_paths, train_labels,
                val_paths, val_labels, 
                num_epochs, criterion, optimizer, scheduler, use_gpu,
                train_img_feats_dict, val_img_feats_dict,
                model_name='',  pkl_file='', epoch_pos=-1,skipped_one=False
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
    global val_acc, val_loss, train_acc, train_loss

    seed = 10
    setup_seed(seed)

    suffix = list(train_img_feats_dict.keys())[0][-3:]

    start_epoch = 0
    if os.path.exists(pkl_file):
        print('pickle file is:', pkl_file)

        net.load_state_dict(torch.load(pkl_file), strict=False)
        
        print("parameters load")

        start_epoch = int(os.path.basename(pkl_file).split('.')[0].split('_')[epoch_pos])

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
    else:
        shutil.rmtree(dir_path, ignore_errors=True)
        os.mkdir(dir_path)


    best_acc = 0.0
    best_model_wts = net.state_dict()
    best_epoch = -1
    since = time.time()

    train_consistent_mean, train_consistent_std = get_consistent_mean(train_paths, train_labels)
    # train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])
    val_consistent_mean, val_consistent_std = get_consistent_mean(val_paths, val_labels)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_consistent_mean, val_consistent_std),
        ]),
    }


    train_data = DatasetWithSingleImage(train_paths, train_labels, data_transforms['train'])
    val_data = DatasetWithSingleImage(val_paths, val_labels, data_transforms['val'])

    img_datasets = {'train': train_data, 'val': val_data}

    # g=torch.Generator()
    # g.manual_seed(10)
    # train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                                 num_workers=1)
    # g.manual_seed(10)
    # val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=20,  drop_last=True, generator=g,
    #                         num_workers=1)

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=5, 
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=5,  
                            num_workers=1)

    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}


    best_epoch = -1
    best_val_acc = -1
    best_train_acc = -1
    best_val_loss = float(inf)
    best_model_wts = net.state_dict()
    
    feats_dict = {'train': train_img_feats_dict, 'val': val_img_feats_dict}

    count = 0


    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    
    for epoch in range(start_epoch, num_epochs):

        running_stages = ['train', 'val']
        # running_stages = ['train']

        print('**'*15, str(epoch) + "starting ", '**' * 15)

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            for j, (paths, inputs, labels) in enumerate(dataloader[phase]):

               
                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()
             
                if len(paths) == 1 and skipped_one:
                    continue

                if phase == 'train':
                    outputs1,outputs2 = net.forward((inputs,))
                else:
                    outputs1 = net.forward((inputs, ))
    
                _, preds = torch.max(outputs1.data, 1)

                if phase == 'train':
                    loss = criterion(outputs1, labels) + criterion(outputs2, labels)
                else:
                    loss = criterion(outputs1, labels)

                for predth in range(preds.shape[0]):
                    preds_list.append(preds[predth].item())
                    labels_list.append(labels[predth].item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                running_loss += loss.data
            
            correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
            uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()

            print('-'*25, '*', '-'*25)

            # print("correct num:", correct_num)
            # print("uncorrect num:", uncorrect_num)
            # print('train datasets length:', (correct_num + uncorrect_num))
            # print('train acc:', correct_num / (correct_num + uncorrect_num))
            
            assert (correct_num + uncorrect_num) == len(labels_list)
            assert len(preds_list) == len(labels_list)

            labels_arr = np.asarray(labels_list)
            preds_arr = np.asarray(preds_list)
            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': preds_arr}
            dict_filename = str(epoch) + '_' + str(phase) + '_dicts.pickle'
            save_dir = dir_path + '/saved_dicts/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)


            save_path = os.path.join(save_dir, dict_filename)
            with open(save_path, 'wb') as fp:
                pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


            if phase == 'train':

                print('-'*25)

                train_loss = running_loss / (len(preds_list) )
                train_acc = correct_num / (len(preds_list) )

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, auc_score))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25, '*', '-'*25)


        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss.item())
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss.item())


        # if val_loss <= best_val_loss:
        if val_acc >= best_val_acc:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_acc = train_acc

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)

            # save_path = dir_path + '/' + \
            # str(model_name) + '_epoch2_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            # torch.save(best_model_wts, save_path)
        else:
            count += 1
        #     if count >= 10:
        #         print("Training Process ends ")
        #         break


    save_path = dir_path + '/' + \
                str(model_name) + '_epoch_' + str(best_epoch) + '_' + str(best_train_acc) + '_' + str(best_val_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)


    loss_and_acc_saved_dir = os.path.join(dir_path, 'lossAndacc')
    if not os.path.exists(loss_and_acc_saved_dir):
        os.mkdir(loss_and_acc_saved_dir)
    else:
        print(loss_and_acc_saved_dir, 'already exists!')


    train_acc_saved_path = os.path.join(loss_and_acc_saved_dir, model_name + '_train_acc.png')
    train_loss_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_train_loss.png')
    # draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    # draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)


    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    # draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    # draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("---------------------train end----------------------")
    return save_path

    

