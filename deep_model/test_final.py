# -*- encoding: utf-8 -*-
"""
@File    : test2.py.py
@Time    : 3/9/2022 6:33 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""

import os
from threading import main_thread
import time
from turtle import mainloop
from unittest.main import MAIN_EXAMPLES
from cv2 import threshold
import pandas as pd
from pip import main
from torch import nn


from classfication_metrics import cal_metrics

from train27 import get_available_data_by_order
from sklearn import metrics

try:
    from PIL import Image
except ImportError:
    print(ImportError)
    import Image

from tqdm import tqdm
from torchvision import datasets, models, transforms
import torch
from torch.utils.data import DataLoader, Dataset
from densenet_plus import Net2, Net4, Net8, Net11
from torch.optim import lr_scheduler
import numpy as np
from feat_tools import *
from __init__ import global_var
import pickle
import torch 
import torch.nn.functional as F


def setup_seed(seed):
	#  下面两个常规设置了，用来np和random的话要设置 
    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
    torch.manual_seed(seed)
    
    torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。
    

device = "cuda:5"
import random


# def worker_init_fn(worked_id):
#     # print("initial seed", torch.initial_seed())
#     # worker_seed = torch.initial_seed() % 2**32
#     worker_seed = 10
#     print("wrok seed:", worker_seed)
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def get_img_feats_dict(feats_csv_file, start_idx=3, suffix='bmp'):

    df = pd.read_csv(feats_csv_file)
    img_feats_dict = dict()
    for index, row in df.iterrows():
        if row['image'] == 'image':
            print("SKIPPED")
            continue
        img_name = row['image'].replace(row['image'][-3:], suffix) if suffix not in row['image'] else row['image']
        img_feats = np.asarray(row[start_idx:], dtype=np.float64)
        img_feats_dict[img_name] = img_feats
    print('len of feats dict:', len(img_feats_dict))
    # print(img_feats_dict)
    return img_feats_dict


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


def get_img_paths_and_labels(dir):
    img_paths, img_labels = [], []

    for img_name in os.listdir(dir):
        img_path = os.path.join(dir, img_name)
        img_paths.append(img_path)
    img_labels = [1 if img_name[0] == 'M' else 0 for img_name in os.listdir(dir)]
    return img_paths, img_labels


def extra_feats_pipeline(img_data, mean_feats_dict, std_feats_dict, return_topo_mask=False, img_size=128):
    # print("img_data type:", type(img_data))
    img_data = np.asarray(img_data)
    assert type(img_data) == np.ndarray
    
    mask_arr = np.ones((img_size, img_size), dtype=np.uint8) * 255
    mask_data = cv2.cvtColor(mask_arr, cv2.COLOR_GRAY2RGB)
   
    # 送入库函数来获取形状和纹理特征
    # print("img shape:", img_data.shape)
    # print("mask shape:", mask_data.shape)
    texture_feats = generate_single_texture_features(img_data, mask_data)

    filtered_feats = []
    # 找到最终选定特征中的所包含的拓扑特征的尺度
    topo_feats_keys = ['Degree', 'Vertex', 'Subgraph', 'Component', 'Average', 'Points', 'Diameter']
    # 根据之前筛选出来的特征关键词来获取有效特征
    feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_with_cv.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_20.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_30.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_9.csv']

    file_idx = 0
    df = pd.read_csv(feats_csv_file_1ist[file_idx])
    df_columns = df.columns.tolist()[3:]

    # print("*" * 30)
    # print("df columns")
    # print(df_columns)
    # print("*" * 30)

    scale_range_set = []

    for filtered_feat in df_columns:
        for key in topo_feats_keys:
            if key in filtered_feat and 'SumAverage' not in filtered_feat:
                # print("key:", key)
                print(filtered_feat)
                scale = filtered_feat.split('_')[-1]
                scale_range_set.append(scale)
                break
    scale_range_set = set(scale_range_set)
    # print('scale_range_set:', scale_range_set)
    scale_range_set = [int(i) for i in scale_range_set]


    topo_feats = []
    threshold = 20
    while topo_feats == []:
        topo_mask_data = get_topo_mask(img_data, threshold=threshold)
        # 送入拓扑特征提取函数获取拓扑特征
        topo_mask_data = topo_mask_data.astype(np.uint8)
        topo_feats = generate_single_topo_features(topo_mask_data, scale_range_set)
        threshold -= 1
    
    # print('--' * 10 + 'topo feats' + '--' * 10)
    # print('threshold is:', threshold)
    # print('--' * 10 + 'topo feats' + '--' * 10)
    # print(topo_feats)
    # print(len(topo_feats))
    # print("topo feats:", topo_feats)

    # # 合并两类特征
    if len(topo_feats) != 0:
        merged_feats = dict(texture_feats, **topo_feats)
    else:
        merged_feats = texture_feats
    # print('--' * 10 + 'merged feats' + '--' * 10)
    # print(len(merged_feats))
    # print(merged_feats) 

    for col in df_columns:
        mean = mean_feats_dict[col]
        std = std_feats_dict[col]
        filtered_feats.append((merged_feats[col] - mean) / (std + 1e-9))
    # print('final feats:', filtered_feats)
    # print(len(filtered_feats))
    filtered_feats = np.asarray(filtered_feats)

    if not return_topo_mask:
        return filtered_feats
    else:
        return filtered_feats, np.asarray(topo_mask_data,np.float32)


def topo_mask_data_pipeline(img_data):
   
    img_data = np.asarray(img_data)
    assert type(img_data) == np.ndarray
 
    filtered_feats = []
    # 找到最终选定特征中的所包含的拓扑特征的尺度
    topo_feats_keys = ['Degree', 'Vertex', 'Subgraph', 'Component', 'Average', 'Points', 'Diameter']
    # 根据之前筛选出来的特征关键词来获取有效特征
    feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_with_cv.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_20.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_30.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_9.csv']

    file_idx = 0
    df = pd.read_csv(feats_csv_file_1ist[file_idx])
    df_columns = df.columns.tolist()[3:]

    scale_range_set = []
    for filtered_feat in df_columns:
        for key in topo_feats_keys:
            if key in filtered_feat and 'SumAverage' not in filtered_feat:
                scale = filtered_feat.split('_')[-1]
                scale_range_set.append(scale)
                break
    scale_range_set = set(scale_range_set)
    # print('scale_range_set:', scale_range_set)
    scale_range_set = [int(i) for i in scale_range_set]

    topo_feats = []
    threshold = 20
    while topo_feats == []:
        topo_mask_data = get_topo_mask(img_data, threshold=threshold)
        # 送入拓扑特征提取函数获取拓扑特征
        topo_mask_data = topo_mask_data.astype(np.uint8)
        topo_feats = generate_single_topo_features(topo_mask_data, scale_range_set)
        threshold -= 1
    
    return np.asarray(topo_mask_data, np.float32)


@torch.no_grad()
def test_model(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDataset(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []

    for j, (paths, inputs, labels, extra_feats) in enumerate(test_dataloader):
        print('extra feats', extra_feats)
        print("inputs shape:", inputs.shape)
        print("extra feats:", extra_feats.shape)

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            net = net.cuda(device)
            # traditional_feats = traditional_feats.cuda(device)

        outputs = net.forward((inputs, extra_feats))
        # outputs = net.forward((inputs, traditional_feats))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))


@torch.no_grad()
def test_model_w_vgg(net, test_dataloader, img_datasets, use_gpu, img_feats_dict, save_path, phase='test', vgg_path=''):


    vgg_model = models.vgg19(pretrained=False)
    fc_list = [vgg_model.classifier[i] for i in range(6)]
    fc_list[-1] = nn.Linear(fc_list[-3].out_features, 1024)
    fc_list.append(nn.ReLU(inplace=True))
    fc_list.append(nn.Dropout(p=0.5, inplace=False))
    fc_list.append(torch.nn.Linear(1024, 2, bias=True))
    vgg_model.classifier = torch.nn.Sequential(*fc_list)
    vgg_model.load_state_dict(torch.load(vgg_path))
    vgg_model.cuda(device)

    net.load_state_dict(torch.load(save_path))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []

    for j, (paths, inputs, labels, extra_feats) in enumerate(test_dataloader):
        print('extra feats', extra_feats)
        print("inputs shape:", inputs.shape)
        print("extra feats:", extra_feats.shape)

        inputs = inputs.cuda(device)
        vgg_feats = vgg_model.features(inputs)
        vgg_feats = vgg_model.avgpool(vgg_feats)
        vgg_feats = vgg_feats.reshape(vgg_feats.size()[0], -1)
        for fc_layer_th in range(len(fc_list) - 1):
            vgg_feats = vgg_model.classifier[fc_layer_th](vgg_feats)

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            net = net.cuda(device)
            vgg_feats = vgg_feats.cuda(device)
            # traditional_feats = traditional_feats.cuda(device)

        outputs = net.forward((inputs, extra_feats, vgg_feats))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

    print('running corrects:', running_corrects)
    test_loss = running_loss / len(img_datasets[phase])
    test_acc = running_corrects / len(img_datasets[phase])
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("correct num:", torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item())
    print("uncorrect num:", torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item())
    print('train datasets length:', len(img_datasets[phase]))
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_consistent_mean_and_std(img_paths, img_labels):
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

    return consistent_mean_sum / len(img_paths), consistent_std_sum / len(img_paths)


def load_img_data_from_path(img_path, loader=pil_loader):
    return loader(img_path)

class TestDataset(Dataset):
    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)

        with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict2.pkl', "rb") as f:
            mean_feats_dict = pickle.load(f)
        with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict2.pkl', "rb") as f:
            std_feats_dict = pickle.load(f)

        extra_feats = extra_feats_pipeline(cur_img_data, mean_feats_dict, std_feats_dict)

        # print('1 in test dataset class extra feats shape:', extra_feats.shape)
        print("Extra Feats!")
        extra_feats = np.asarray(extra_feats)
        extra_feats = torch.Tensor(extra_feats)

        # print('2 in test dataset class extra feats shape:', extra_feats.shape)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, cur_img_data, cur_img_label, extra_feats


class TestDatasetWithMaskData(Dataset):
    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)

        with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict2.pkl', "rb") as f:
            mean_feats_dict = pickle.load(f)
        with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict2.pkl', "rb") as f:
            std_feats_dict = pickle.load(f)

        extra_feats, topo_mask_data = extra_feats_pipeline(cur_img_data, mean_feats_dict, std_feats_dict, True)
        topo_mask_data = topo_mask_data.reshape(3, topo_mask_data.shape[0], topo_mask_data.shape[1])

        print('10'*30)
        print("Extra Feats!")
        extra_feats = np.asarray(extra_feats)
        print(extra_feats)
        print('10'*30)
        print('\n\n')

        extra_feats = torch.Tensor(extra_feats)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, cur_img_data, cur_img_label, extra_feats, topo_mask_data


class TestDatasetWithSingleImage(Dataset):
    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, cur_img_data, cur_img_label
   


class TestDatasetWithMaskData2(Dataset):
    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)
        cur_img_data_arr = np.asarray(cur_img_data)

        with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict2.pkl', "rb") as f:
            mean_feats_dict = pickle.load(f)
        with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict2.pkl', "rb") as f:
            std_feats_dict = pickle.load(f)

        extra_feats, topo_mask_data = extra_feats_pipeline(cur_img_data, mean_feats_dict, std_feats_dict, True)
        topo_mask_data_arr = cv2.cvtColor(topo_mask_data, cv2.COLOR_BGR2GRAY).reshape((1, cur_img_data_arr.shape[0],
                     cur_img_data_arr.shape[1]))
        topo_mask_data_arr_tensor = torch.Tensor(topo_mask_data_arr)

        topo_mask_data = topo_mask_data.reshape(3, topo_mask_data.shape[0], topo_mask_data.shape[1])

        print('10'*30)
        print("Extra Feats!")
        extra_feats = np.asarray(extra_feats)
        print(extra_feats)
        print('10'*30)
        print('\n\n')

        extra_feats = torch.Tensor(extra_feats)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
                stacked_img_data = torch.cat((cur_img_data, topo_mask_data_arr_tensor), 0)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, stacked_img_data, cur_img_label, extra_feats, topo_mask_data


class TestDatasetWithMaskData3(Dataset):
    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)
        cur_img_data_arr = np.asarray(cur_img_data)

        with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict2.pkl', "rb") as f:
            mean_feats_dict = pickle.load(f)
        with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict2.pkl', "rb") as f:
            std_feats_dict = pickle.load(f)

        extra_feats, topo_mask_data = extra_feats_pipeline(cur_img_data, mean_feats_dict, std_feats_dict, True)
        topo_mask_data_arr = cv2.cvtColor(topo_mask_data, cv2.COLOR_BGR2GRAY).reshape((1, cur_img_data_arr.shape[0],
                     cur_img_data_arr.shape[1]))
        topo_mask_data_arr[np.where(topo_mask_data_arr!=0)] = 1
        if len( np.where( (topo_mask_data_arr!=0) & (topo_mask_data_arr!=1) )[0] ):
            print("-error-" * 10)
            print('TOPO MASK ERROR')
            print("-error-" * 10)
        assert np.min(topo_mask_data_arr) == 0 and np.max(topo_mask_data_arr) == 1

        topo_mask_data_arr_tensor = torch.Tensor(topo_mask_data_arr)

        topo_mask_data = topo_mask_data.reshape(3, topo_mask_data.shape[0], topo_mask_data.shape[1])

        print('10'*30)
        print("Extra Feats!")
        extra_feats = np.asarray(extra_feats)
        print(extra_feats)
        print('10'*30)
        print('\n\n')

        extra_feats = torch.Tensor(extra_feats)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
                added_img_data = torch.add(cur_img_data * 0.7, topo_mask_data_arr_tensor * 0.3)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, added_img_data, cur_img_label, extra_feats, topo_mask_data


def thresholding_img(img_arr, threshold=165):
    if len(img_arr.shape) == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    changed_img_arr = np.zeros_like(img_arr)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if img_arr[i][j] >= threshold:
                changed_img_arr[i][j] = img_arr[i][j]
    changed_img_arr = cv2.cvtColor(changed_img_arr, cv2.COLOR_GRAY2BGR)
    return changed_img_arr

class TestDatasetWithThresholdingData(Dataset):
    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)
        cur_img_data = np.asarray(cur_img_data)
        cur_img_data = thresholding_img(cur_img_data)

        with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict2.pkl', "rb") as f:
            mean_feats_dict = pickle.load(f)
        with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict2.pkl', "rb") as f:
            std_feats_dict = pickle.load(f)

        extra_feats, topo_mask_data = extra_feats_pipeline(cur_img_data, mean_feats_dict, std_feats_dict, True)
        topo_mask_data = topo_mask_data.reshape(3, topo_mask_data.shape[0], topo_mask_data.shape[1])

        print('10'*30)
        print("Extra Feats!")
        extra_feats = np.asarray(extra_feats)
        print(extra_feats)
        print('10'*30)
        print('\n\n')

        extra_feats = torch.Tensor(extra_feats)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, cur_img_data, cur_img_label, extra_feats, topo_mask_data


class DatasetWithMaskData(Dataset):
    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        # print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)

        topo_mask_data = topo_mask_data_pipeline(cur_img_data)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        cur_img_name = os.path.basename(cur_img_path)
        return cur_img_name, cur_img_data, cur_img_label, topo_mask_data


class DatasetWithMaskDataOffline(Dataset):

    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        # print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)

        cur_img_name = os.path.basename(cur_img_path)
        
        topo_mask_path = os.path.join( '/'.join(os.path.dirname(cur_img_path).split('/')[:-1]) + '/topo_mask/', cur_img_name[:-4] + '_mask.jpg')
        topo_mask_data = load_img_data_from_path(topo_mask_path)
        topo_mask_data =  transforms.ToTensor()(topo_mask_data)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        return cur_img_name, cur_img_data, cur_img_label, topo_mask_data



class DatasetWithSingleImage(Dataset):

    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        # print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)

        cur_img_name = os.path.basename(cur_img_path)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        return cur_img_name, cur_img_data, cur_img_label




class DatasetWithMaskDataOnly(Dataset):

    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        cur_img_name = os.path.basename(cur_img_path)
        
        topo_mask_path = os.path.join( '/'.join(os.path.dirname(cur_img_path).split('/')[:-1]) + '/topo_mask/', cur_img_name[:-4] + '_mask.jpg')
        topo_mask_data = load_img_data_from_path(topo_mask_path)
        topo_mask_data =  transforms.ToTensor()(topo_mask_data)

        return cur_img_name, cur_img_label, topo_mask_data


class DatasetWithStackedData(Dataset):

    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        # print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)
        cur_img_data_arr = np.asarray(cur_img_data)

        cur_img_name = os.path.basename(cur_img_path)
        
        topo_mask_path = os.path.join( '/'.join(os.path.dirname(cur_img_path).split('/')[:-1]) + '/topo_mask/', cur_img_name[:-4] + '_mask.jpg')
        topo_mask_data = load_img_data_from_path(topo_mask_path)
        topo_mask_data_arr = np.asarray(topo_mask_data)
        topo_mask_data_arr = cv2.cvtColor(topo_mask_data_arr, cv2.COLOR_BGR2GRAY).reshape((1, cur_img_data_arr.shape[0],
                     cur_img_data_arr.shape[1]))
        topo_mask_data_arr_tensor = torch.Tensor(topo_mask_data_arr)

        topo_mask_data =  transforms.ToTensor()(topo_mask_data)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
                stacked_img_data = torch.cat((cur_img_data, topo_mask_data_arr_tensor), 0)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        return cur_img_name, stacked_img_data, cur_img_label, topo_mask_data


class DatasetWithAddedData(Dataset):

    def __init__(self, test_img_path, test_img_label, data_transforms=None):
        self.test_img_path = test_img_path
        self.test_img_label = test_img_label
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.test_img_path)

    def __getitem__(self, item):
        cur_img_path = self.test_img_path[item]
        cur_img_label = self.test_img_label[item]

        # print('cur img path is:', cur_img_path)
        cur_img_data = load_img_data_from_path(cur_img_path)
        cur_img_data_arr = np.asarray(cur_img_data)

        cur_img_name = os.path.basename(cur_img_path)
        
        topo_mask_path = os.path.join( '/'.join(os.path.dirname(cur_img_path).split('/')[:-1]) + '/topo_mask/', cur_img_name[:-4] + '_mask.jpg')
        topo_mask_data = load_img_data_from_path(topo_mask_path)
        topo_mask_data_arr = np.asarray(topo_mask_data)
        topo_mask_data_arr = cv2.cvtColor(topo_mask_data_arr, cv2.COLOR_BGR2GRAY).reshape((1, cur_img_data_arr.shape[0],
                     cur_img_data_arr.shape[1]))
        topo_mask_data_arr[np.where(topo_mask_data_arr != 0)] = 1
        if len(np.where((topo_mask_data_arr != 0) & (topo_mask_data_arr != 1))[0]):
            print("-error-" * 10)
            print("TOPO MASK ERROR")
            print("-error-" * 10)
        assert np.min(topo_mask_data_arr) == 0 and np.max(topo_mask_data_arr) == 1

        topo_mask_data_arr_tensor = torch.Tensor(topo_mask_data_arr)

        topo_mask_data =  transforms.ToTensor()(topo_mask_data)

        if self.data_transforms is not None:
            try:
                cur_img_data = self.data_transforms(cur_img_data)
                added_img_data = torch.add(cur_img_data * 0.7, topo_mask_data_arr_tensor * 0.3)
            except Exception as e:
                print(e)
                print(cur_img_path, " cannot make transform!")

        return cur_img_name, added_img_data, cur_img_label, topo_mask_data






@torch.no_grad()
def test_model_w_topo_mask_online(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1):

    seed = 10
    setup_seed(seed)

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    g = torch.Generator()
    g.manual_seed(seed)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1, generator=g,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path, map_location=device))

    # net.train(True)
    net.train(False)
    # net.eval()


    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

       
        outputs = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        # preds = 1 - preds
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)

        loss = criterion(outputs, labels)

        print("the labels is:", labels.data)

        labels_list.append(labels.data[0].item())

        paths_list.append(paths)

        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    print("labels type:", type(labels_list))
    print("preds type:", type(preds_list))

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = correct_num / (correct_num + uncorrect_num)
    
    labels_arr = np.asarray(labels_list)
    preds_arr = np.asarray(preds_list)

    dict = {'labels': labels_arr, 'preds': preds_arr, }
    dict_filename = str(saved_epoch) + '_test_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts/'
    save_path = os.path.join(save_dir, dict_filename)

    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, auc_score))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))




@torch.no_grad()
def test_model_w_topo_mask_online3(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1):

    seed = 10
    setup_seed(seed)

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithSingleImage(test_img_paths, test_img_labels, data_transforms['test'])
    g = torch.Generator()
    g.manual_seed(seed)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1, generator=g,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path, map_location=device))

    # net.train(True)
    net.train(False)
    # net.eval()


    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            net = net.cuda(device)

       
        outputs = net.forward((inputs,))

        _, preds = torch.max(outputs.data, 1)
        # preds = 1 - preds
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)

        loss = criterion(outputs, labels)

        print("the labels is:", labels.data)

        labels_list.append(labels.data[0].item())

        paths_list.append(paths)

        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    print("labels type:", type(labels_list))
    print("preds type:", type(preds_list))

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = correct_num / (correct_num + uncorrect_num)
    
    labels_arr = np.asarray(labels_list)
    preds_arr = np.asarray(preds_list)

    dict = {'labels': labels_arr, 'preds': preds_arr, }
    dict_filename = str(saved_epoch) + '_test_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts/'
    save_path = os.path.join(save_dir, dict_filename)

    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, auc_score))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))



    

@torch.no_grad()
def test_model_w_topo_mask_online2(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1, w_loss=1.1):

    seed = 10
    setup_seed(seed)

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    g = torch.Generator()
    g.manual_seed(seed)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1, generator=g,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path, map_location=device))

    # net.train(True)
    net.train(False)
    # net.eval()


    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

       
        outputs, outputs_1, outputs_2 = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        # preds = 1 - preds
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)

        loss = w_loss * criterion(outputs, labels) +  criterion(outputs_1, labels) +  criterion(outputs_2, labels)

        print("the labels is:", labels.data)

        labels_list.append(labels.data[0].item())

        paths_list.append(paths)

        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    print("labels type:", type(labels_list))
    print("preds type:", type(preds_list))

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = correct_num / (correct_num + uncorrect_num)
    
    labels_arr = np.asarray(labels_list)
    preds_arr = np.asarray(preds_list)

    dict = {'labels': labels_arr, 'preds': preds_arr, }
    dict_filename = str(saved_epoch) + '_test_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts/'
    save_path = os.path.join(save_dir, dict_filename)

    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, auc_score))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))



@torch.no_grad()
def test_model_w_topo_mask_online21(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1, w_loss=1.1):

    seed = 10
    setup_seed(seed)

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    g = torch.Generator()
    g.manual_seed(seed)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1, generator=g,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path, map_location=device))

    net.train(False)

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

       
        outputs, outputs_1, outputs_2, weights = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        # preds = 1 - preds
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)

        loss = w_loss * criterion(outputs, labels) +  criterion(outputs_1, labels) +  criterion(outputs_2, labels)

        print("the labels is:", labels.data)

        labels_list.append(labels.data[0].item())

        paths_list.append(paths)

        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    print("labels type:", type(labels_list))
    print("preds type:", type(preds_list))

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = correct_num / (correct_num + uncorrect_num)
    
    labels_arr = np.asarray(labels_list)
    preds_arr = np.asarray(preds_list)

    dict = {'labels': labels_arr, 'preds': preds_arr, }
    dict_filename = str(saved_epoch) + '_test_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts/'
    save_path = os.path.join(save_dir, dict_filename)

    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, auc_score))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))



def test_feats():
    cur_img_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_aug_4_times/roi/M607.jpg'
    cur_img_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_aug_4_times/roi/B608.jpg'
    cur_img_path = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_aug_4_times/roi/M65LMLO.jpg'
    cur_img_label = 1

    print('cur img path is:', cur_img_path)
    cur_img_data = load_img_data_from_path(cur_img_path)

    with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict2.pkl', "rb") as f:
        mean_feats_dict = pickle.load(f)
    with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict2.pkl', "rb") as f:
        std_feats_dict = pickle.load(f)

    extra_feats, topo_mask_data = extra_feats_pipeline(cur_img_data, mean_feats_dict, std_feats_dict, True)

    print("Extra Feats!")
    print(extra_feats)
    extra_feats = np.asarray(extra_feats)
    extra_feats = torch.Tensor(extra_feats)

    # print('2 in test dataset class extra feats shape:', extra_feats.shape)


# if __name__ == "__main__":

#     test_feats()


def test_model2(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)

    # test_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    # test_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDataset(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    net.train(True)
    # net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []

    for j, (paths, inputs, labels, extra_feats) in enumerate(test_dataloader):
        print('extra feats', extra_feats)
        print("inputs shape:", inputs.shape)
        print("extra feats:", extra_feats.shape)

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            net = net.cuda(device)
            # traditional_feats = traditional_feats.cuda(device)

        outputs = net.forward((inputs, extra_feats))
        print("outputs:", outputs)
        # outputs = net.forward((inputs, traditional_feats))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))


@torch.no_grad()
def test_model_w_topo_mask_online_2(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    malignant_mis_classified = []
    benign_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

        outputs = net.forward((extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))



@torch.no_grad()
def test_model_w_topo_mask_online_3(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    print("testing model ...")

    
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)

    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithThresholdingData(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):


        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

        outputs = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))




@torch.no_grad()
def test_model_w_topo_mask_online_4(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData2(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

        outputs = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))


@torch.no_grad()
def test_model_w_topo_mask_online_5(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData3(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

        outputs = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))



@torch.no_grad()
def test_model_with_topo_model(net, topo_model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

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

        outputs = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))


@torch.no_grad()
def test_model_with_topo_model(net, topo_model, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

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

        outputs = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))



@torch.no_grad()
def test_topo_model(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test'):

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path))

    net.train(False)

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

        outputs = net.forward(topo_mask_data)

        _, preds = torch.max(outputs.data, 1)
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)
        loss = criterion(outputs, labels)
        print("the labels is:", labels.data)
        labels_list.append(labels.data[0].item())
        paths_list.append(paths)
        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = running_corrects / (correct_num + uncorrect_num)
    
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))




##################################################
#
#  改正其中的auc计算
#
###################################################
@torch.no_grad()
def test_model_w_topo_mask_online_corrected(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1):

    seed = 10
    setup_seed(seed)

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    g = torch.Generator()
    g.manual_seed(seed)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1, generator=g,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path, map_location=device))

    # net.train(True)
    net.train(False)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

       
        outputs = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        # preds = 1 - preds
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)

        loss = criterion(outputs, labels)

        print("the labels is:", labels.data)

        labels_list.append(labels.data[0].item())

        paths_list.append(paths)

        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    print("labels type:", type(labels_list))
    print("preds type:", type(preds_list))

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = correct_num / (correct_num + uncorrect_num)
    
    labels_arr = np.asarray(labels_list)
    preds_arr = np.asarray(preds_list)

    dict = {'labels': labels_arr, 'preds': preds_arr, }
    dict_filename = str(saved_epoch) + '_test_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts/'
    save_path = os.path.join(save_dir, dict_filename)

    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, auc_score))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))



#####################################################################################
#
#  改正auc的计算方法
#  1. 将预测结果改成概率向量
#
######################################################################################



@torch.no_grad()
def test_model_w_correct_auc(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1):

    seed = 10
    setup_seed(seed)

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    g = torch.Generator()
    g.manual_seed(seed)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1, generator=g,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path, map_location=device))

    # net.train(True)
    # if phase == 'test':
    #     net.train(False)
    # elif phase == 'val':
    #     # net.train(True)
    #     net.train(False)
    # else:
    #     pass
    net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []
    benign_mis_classified = []
    malignant_mis_classified = []

    malignant_pred_probs_tensor = None

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            extra_feats = extra_feats.cuda(device)
            topo_mask_data = topo_mask_data.cuda(device)
            net = net.cuda(device)

       
        outputs = net.forward((inputs, extra_feats, topo_mask_data))

        _, preds = torch.max(outputs.data, 1)
        # preds = 1 - preds
        preds_list.append(preds.data[0].item())
        print("the preds is:", preds.data)

        loss = criterion(outputs, labels)

        print("the labels is:", labels.data)

        labels_list.append(labels.data[0].item())

        paths_list.append(paths)

        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

        # outputs_sf = F.softmax(F.sigmoid(outputs), -1)
        outputs_sf = F.softmax(outputs, -1)
        if j == 0:
            malignant_pred_probs_tensor = outputs_sf[:, 1]
        else:
            malignant_pred_probs_tensor = torch.cat((malignant_pred_probs_tensor, outputs_sf[:, 1]), 0)

    print('running corrects:', running_corrects)
 
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("malignant_mis_classified:", malignant_mis_classified)
    print("benign_mis_classified:", benign_mis_classified)

    print("labels type:", type(labels_list))
    print("preds type:", type(preds_list))

    correct_num = torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item()
    uncorrect_num = torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item()
    print("correct num:", correct_num)
    print("uncorrect num:", uncorrect_num)
    print('train datasets length:', (correct_num + uncorrect_num))
    
    test_loss = running_loss / (correct_num + uncorrect_num)
    test_acc = correct_num / (correct_num + uncorrect_num)
    
    labels_arr = np.asarray(labels_list)
    preds_arr = np.asarray(preds_list)

    pred_probs_arr = malignant_pred_probs_tensor.cpu().detach().numpy()
    fpr, tpr, threshold = metrics.roc_curve(labels_arr, pred_probs_arr)

    correct_auc_1 = metrics.roc_auc_score(labels_arr, pred_probs_arr)

    correct_auc_2 = metrics.auc(fpr, tpr)

    assert correct_auc_1 == correct_auc_2

    dict = {'labels': labels_arr, 'preds': pred_probs_arr}
    dict_filename = str(saved_epoch) + '_' + phase + '_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts2/'
    save_path = os.path.join(save_dir, dict_filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, correct_auc_1))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

