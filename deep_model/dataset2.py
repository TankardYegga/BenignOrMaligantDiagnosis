# -*- encoding: utf-8 -*-
"""
@File    : dataset.py
@Time    : 11/30/2021 12:58 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
try:
    from PIL import Image
except ImportError:
    print(ImportError)
    import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    #     return pil_loader(path)
    return pil_loader(path)


class CustomData(Dataset):
    def __init__(self, img_path, dataset_split_name='', data_transforms=None, loader=default_loader):
        self.data_transforms = data_transforms
        self.dataset_split_name = dataset_split_name
        self.loader = loader

        self.images = []
        self.labels = []
        benign_dir = os.path.join(img_path, 'Benign')
        maligant_dir = os.path.join(img_path, 'Malignant')
        for benign_img in os.listdir(benign_dir):
            self.images.append(os.path.join(benign_dir, benign_img))
            self.labels.append(0)
        for maligant_img in os.listdir(maligant_dir):
            self.images.append(os.path.join(maligant_dir, maligant_img))
            self.labels.append(1)

        # for i in range(len(self.images)):
        #     print(self.images[i])
        #     print(self.labels[i])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        single_img_path = self.images[item]
        single_img_name = os.path.basename(single_img_path)
        single_img = self.loader(single_img_path)
        label = self.labels[item]

        if self.data_transforms is not None:
            try:
                single_img = self.data_transforms[self.dataset_split_name](single_img)
            except Exception as e:
                print(e)
                print(single_img_path, " cannot make transform!")

        return single_img_name, single_img, label


# if __name__ == '__main__':
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
    # csv_path = r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\filtered_features.csv'
    # image_datasets = {x: CustomData(img_path='D:/AllExploreDownloads/IDM/ExtraCode/split_data/' + x,
    #                                 # csv_path=csv_path,
    #                                 data_transforms=data_transforms,
    #                                 dataset_split_name=x) for x in ['train', 'val']}
    # print(len(image_datasets['train']))
    # print(len(image_datasets['val']))
    # print(image_datasets['train'][341])
    # train_data_loaders = DataLoader(image_datasets['train'], shuffle=True,  batch_size=10,
    #                                 num_workers=1)
    # val_data_loaders = DataLoader(image_datasets['val'], shuffle=True, batch_size=10,
    #                               num_workers=5)
    # image_dataloaders = {'train': train_data_loaders, 'val': val_data_loaders}
    # for data in train_data_loaders:
    #     images, labels = data
    #     print(images.shape)
    #     print(labels.shape)

    # train_data_loaders_iter = iter(train_data_loaders)
    # while 1:
    #     try:
    #         images, labels = next(train_data_loaders_iter)
    #         print('images', images.shape)
    #         print('labels', labels.shape)
    #     except Exception as e:
    #         print('e')