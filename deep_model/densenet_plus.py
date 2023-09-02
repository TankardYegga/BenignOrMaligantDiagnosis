# -*- encoding: utf-8 -*-
"""
@File    : densenet_plus.py
@Time    : 12/31/2021 6:51 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
from bdb import effective
from copy import deepcopy
import math
from multiprocessing.sharedctypes import Value
from operator import mod
from re import S, T
from time import sleep
from tkinter.tix import Tree
from turtle import forward
from importlib_metadata import requires
from jinja2 import pass_environment
from numpy import isin
from sklearn.semi_supervised import SelfTrainingClassifier
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet_ImageNet import ACmix_ResNet
from ResNet_ImageNet2 import ACmix_ResNet2
from extra_models.ReductionCell import ReductionCell
from swin_transformer import SwinTransformer
from trfm import *
from pool_trfm import *
from torch.autograd import gradcheck

class Net(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net, self).__init__()
        # -2表示去掉model的后两层
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(1024 + num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(num_mid_feats, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        # print(deep_feats.size())
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())
        merged_feats = torch.cat([deep_feats, x[1]], axis=1)

        # 将一个多行的Tensor,拼接成一行,-1指在不告诉函数有多少列
        merged_feats = merged_feats.view(merged_feats.size(0), -1)
        fc1 = self.fc1(merged_feats)
        fc2 = self.fc2(fc1)
        return fc2


class Net2(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_classes=2):
        super(Net2, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1024 + num_mid_feats, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 把densenet121改变成resnet50
class Net15(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_classes=2):
        super(Net15, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 把densenet121改变成resnet50, 并增加trfm
class Net17(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net17, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        # self.dp = nn.Dropout(p=0.2, inplace=False)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        # fc2 = self.dp(fc2)
        return fc2



class Net17Simple(nn.Module):
    def __init__(self, model, densenet_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net17Simple, self).__init__()
        self.res_layer = model
        self.densenet_layer = densenet_model.res_layer
        
        self.fc = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        res_feats = self.res_layer(x[0])

        dense_feats = self.densenet_layer(x[0])

        stacked_feats = torch.stack((res_feats, dense_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        res_feats, dense_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([res_feats, dense_feats], axis=1)
      
        fc = self.fc(merged_feats)
        return fc



class Net17SimplePretrained(nn.Module):
    def __init__(self, model, densenet_model, num_mid_feats=1000, num_classes=2):
        super(Net17SimplePretrained, self).__init__()
        self.res_layer = model.model
        self.densenet_layer = densenet_model.res_layer
        
        self.fc = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        res_feats = self.res_layer(x[0])

        dense_feats = self.densenet_layer(x[0])

        stacked_feats = torch.stack((res_feats, dense_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        res_feats, dense_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([res_feats, dense_feats], axis=1)
      
        fc = self.fc(merged_feats)
        return fc



# 在简化的模型上增加deep——latent模块
class Net17SimplePretrained_W_DeepLatent(nn.Module):
    def __init__(self, model, densenet_model, num_mid_feats=1000, num_classes=2, latent_from_dim=10, latent_to_dim=100):
        super(Net17SimplePretrained_W_DeepLatent, self).__init__()
        self.res_layer = model.model
        self.densenet_layer = densenet_model.res_layer
        

        self.deep_latent = torch.nn.Parameter(torch.randn(latent_from_dim), requires_grad=True)
        self.deep_latent_fc = nn.Linear(latent_from_dim, latent_to_dim)


        self.fc = nn.Linear(1000 + num_mid_feats + latent_to_dim, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)


    def forward(self, x):
        res_feats = self.res_layer(x[0])

        dense_feats = self.densenet_layer(x[0])

        stacked_feats = torch.stack((res_feats, dense_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        res_feats, dense_feats = trfmed_feats.unbind(0)

        deep_latent_embed = self.deep_latent_fc(self.deep_latent)
        deep_latent_embed = deep_latent_embed.repeat(res_feats.shape[0], 1)
        merged_feats = torch.cat([res_feats, dense_feats, deep_latent_embed], axis=1)
      
        fc = self.fc(merged_feats)
        return fc



class Net17Simple2(nn.Module):
    def __init__(self, model1, model2, num_classes=2):
        super(Net17Simple2, self).__init__()
        self.res_layer = model1.res_layer
        self.densenet_layer = model2.res_layer
        
        self.fc = nn.Linear(1000, num_classes)
     

    def forward(self, x):
        res_feats = self.res_layer(x[0])

        dense_feats = self.densenet_layer(x[0])

        merged_feats = torch.add(res_feats, dense_feats) / 2
      
        fc = self.fc(merged_feats)
        return fc


class Net17Simple3(nn.Module):
    def __init__(self, res_model, densenet_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net17Simple3, self).__init__()
        self.res_layer = res_model.res_layer
        self.densenet_layer = densenet_model.res_layer
        
        self.fc = nn.Linear(1000, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        res_feats = self.res_layer(x[0])

        dense_feats = self.densenet_layer(x[0])

        stacked_feats = torch.stack((res_feats, dense_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        res_feats, dense_feats = trfmed_feats.unbind(0)
        merged_feats = torch.add(res_feats, dense_feats) / 2
      
        fc = self.fc(merged_feats)
        return fc


class Net17Plus(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net17Plus, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats + 256, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)

        topo_feats = x[2]
        merged_feats = torch.cat([deep_feats, extra_feats, topo_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net17PlusPlus(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net17PlusPlus, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats + 1000, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        topo_feats = x[2]

        stacked_feats = torch.stack((deep_feats, extra_feats, topo_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats, topo_feats = trfmed_feats.unbind(0)

        merged_feats = torch.cat([deep_feats, extra_feats, topo_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net17PlusPlusPlus(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net17PlusPlusPlus, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats + 1000, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        topo_feats = x[2]
       
        merged_feats = torch.cat([deep_feats * 0.7, extra_feats * 0.2, topo_feats * 0.1], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net17PlusPlusPlusPlus(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net17PlusPlusPlusPlus, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc3 = nn.Linear(1000, 1000)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats + 1000, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        topo_feats = x[2]
        topo_feats = self.fc3(topo_feats)
       
        merged_feats = torch.cat([deep_feats * 0.7, extra_feats * 0.2, topo_feats * 0.1], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 把densenet121改变成resnet50, 并增加trfm和dropout
class Net21(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net21, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.dropout(self.fc2(merged_feats))
        return fc2


# 把densenet121改变成resnet50, 并增加trfm
class Net22(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net22, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = SmallWindowAttention(dim=num_mid_feats, num_heads=2)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 把densenet121改变成resnet50,并增加trfm和多层fc
class Net18(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net18, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        fc3 = self.fc3(fc2)
        return fc3


# 两个resnet50分别用于原图片和topo_mask, 
class Net19(nn.Module):
    def __init__(self, model1, model2, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net19, self).__init__()
        self.img_res_layer = model1
        self.topo_mask_res_layer = model2
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(2000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.img_res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        topo_feats = self.topo_mask_res_layer(x[2])

        stacked_feats = torch.stack((deep_feats, extra_feats, topo_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats, topo_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats, topo_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# resnet50只用于topo_mask
class Net20(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net20, self).__init__()
        self.topo_mask_res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        extra_feats = x[0]
        extra_feats = self.fc1(extra_feats)

        topo_feats = self.topo_mask_res_layer(x[1])
        stacked_feats = torch.stack((extra_feats, topo_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        extra_feats, topo_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([extra_feats, topo_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 把densenet121替换成swin-transformer
class Net16(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_classes=2):
        super(Net16, self).__init__()
        self.swin_t_base = model
                                
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1024 + num_mid_feats, num_classes)

    def forward(self, x):
        deep_feats = self.swin_t_base(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net3(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_classes=2):
        super(Net3, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print(extra_feats.size())
        merged_feats = torch.add(deep_feats, extra_feats)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net4(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_classes=2, num_heads=4):
        super(Net4, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1024 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=num_heads)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        
        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)

        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 增加dropout
class Net13(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_classes=2):
        super(Net13, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)

        self.fc2 = nn.Linear(1024 + num_mid_feats, 1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        
        self.fc3 = nn.Linear(1024, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=0.5, inplace=False)

        self.fc4 = nn.Linear(256, num_classes)
        
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=4)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.dropout1(self.relu1(self.fc1(extra_feats)))
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        
        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)

        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.dropout2(self.relu2(self.fc2(merged_feats)))
        fc3 = self.dropout3(self.relu3(self.fc3(fc2)))
        fc4 = self.fc4(fc3)
        return fc4


# 增加dropout + 使用2层trfm
class Net14(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_classes=2):
        super(Net14, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)

        self.fc2 = nn.Linear(1024 + num_mid_feats, 1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        
        self.fc3 = nn.Linear(1024, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=0.5, inplace=False)

        self.fc4 = nn.Linear(256, num_classes)
        
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=4)
        self.trfm2 = WindowAttention(dim=num_mid_feats, num_heads=4)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        extra_feats = self.dropout1(self.relu1(extra_feats))
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        
        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)
        deep_feats, extra_feats = trfmed_feats.unbind(0)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm2(stacked_feats)
        deep_feats, extra_feats = trfmed_feats.unbind(0)

        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.dropout2(self.relu2(self.fc2(merged_feats)))
        fc3 = self.dropout3(self.relu3(self.fc3(fc2)))
        fc4 = self.fc4(fc3)
        return fc4


# densenet121 + vgg19 + extra_info + concat_fusion
class Net5(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_vgg_feats=1024, num_classes=2):
        super(Net5, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1024 + num_mid_feats + num_vgg_feats, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        vgg_feats = x[2]
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats, vgg_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# densenet121 + vgg19 + extra_info + trfm_fusion
class Net6(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_vgg_feats=1024, num_classes=2):
        super(Net6, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1024 + num_mid_feats + num_vgg_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=4)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())

        vgg_feats = x[2]

        stacked_feats = torch.stack((deep_feats, extra_feats, vgg_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats, vgg_feats = trfmed_feats.unbind(0)

        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats, vgg_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# densenet121 + vgg19 + extra_info + trfm_fusion + multi_fc
class Net7(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_vgg_feats=1024, num_classes=2):
        super(Net7, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1024 + num_mid_feats + num_vgg_feats, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=4)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())

        vgg_feats = x[2]
        stacked_feats = torch.stack((deep_feats, extra_feats, vgg_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats, vgg_feats = trfmed_feats.unbind(0)

        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats, vgg_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)
        return fc4


# densenet121 + vgg19 + extra_info + concat_fusion + multi_fc
class Net8(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_vgg_feats=1024, num_classes=2):
        super(Net8, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1024 + num_mid_feats + num_vgg_feats, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        vgg_feats = x[2]
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats, vgg_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)
        return fc4


# densenet121 + extra_info + trfm_fusion + multi_fc + dropout_fc
class Net9(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_classes=2):
        super(Net9, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)

        self.fc2 = nn.Linear(1024 + num_mid_feats, 512)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)

        self.fc3 = nn.Linear(512, num_classes)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=0.5, inplace=False)

        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=4)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        extra_feats = self.dropout1(self.relu1(extra_feats))

        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)

        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        fc2 = self.dropout2(self.relu2(fc2))
        fc3 = self.fc3(fc2)
        fc3 = self.dropout3(self.relu3(fc3))
        return fc3


# densenet121 + vgg19 + extra_info + trfm_fusion + multi_fc + dense_attention
class Net10(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_vgg_feats=1024, num_classes=2):
        super(Net10, self).__init__()
        feature_layers = list(model.children())[0]
        top_layers_list = [
            feature_layers.conv0, feature_layers.norm0, feature_layers.relu0, feature_layers.pool0, 
            feature_layers.denseblock1, feature_layers.transition1, feature_layers.denseblock2, feature_layers.transition2,
            feature_layers.denseblock3, feature_layers.transition3
        ]

        self.top_layers = nn.Sequential(*top_layers_list)
        self.denseblock4 = feature_layers.denseblock4
        self.pool_trfm_branch = PoolFormerBlock(dim=512)

        self.fusion_conv = nn.Conv2d(kernel_size=1, in_channels=1536, out_channels=1024)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1024 + num_mid_feats + num_vgg_feats, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=4)


    def forward(self, x):
        deep_feats = self.top_layers(x[0])

        trfm_branch = self.pool_trfm_branch(deep_feats)

        deep_feats = torch.concat([self.denseblock4(deep_feats), trfm_branch], axis=1)

        deep_feats = self.fusion_conv(deep_feats)

        deep_feats = F.relu(deep_feats, inplace=True)

        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())

        vgg_feats = x[2]
        stacked_feats = torch.stack((deep_feats, extra_feats, vgg_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats, vgg_feats = trfmed_feats.unbind(0)

        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats, vgg_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)
        return fc4

# Net2 + dropout + 多个fc层
class Net11(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_classes=2):
        super(Net11, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-1])

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)

        self.fc2 = nn.Linear(1024 + num_mid_feats, 1024) 
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)

        self.fc3 = nn.Linear(1024, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=0.5, inplace=False)

        self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = F.relu(deep_feats, inplace=True)
        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        extra_feats = self.dropout1(self.relu1(extra_feats))
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())
        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.dropout2(self.relu2(self.fc2(merged_feats)))
        fc3 = self.dropout3(self.relu3(self.fc3(fc2)))
        fc4 = self.fc4(fc3)

        return fc4


# densenet121 + vgg19(not pretrained) + extra_info + trfm_fusion + multi_fc + dense_attention
class Net12(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1024, num_vgg_feats=1024, num_classes=2):
        super(Net12, self).__init__()
        feature_layers = list(model.children())[0]
        top_layers_list = [
            feature_layers.conv0, feature_layers.norm0, feature_layers.relu0, feature_layers.pool0, 
            feature_layers.denseblock1, feature_layers.transition1, feature_layers.denseblock2, feature_layers.transition2,
            feature_layers.denseblock3, feature_layers.transition3
        ]

        self.top_layers = nn.Sequential(*top_layers_list)
        self.denseblock4 = feature_layers.denseblock4
        self.pool_trfm_branch = PoolFormerBlock(dim=512)

        self.fusion_conv = nn.Conv2d(kernel_size=1, in_channels=1536, out_channels=1024)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1024 + num_mid_feats + num_vgg_feats, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=4)

        vgg_model = models.vgg19(pretrained=False)
        fc_list = [vgg_model.classifier[i] for i in range(6)]
        fc_list.append(nn.Linear(fc_list[-3].out_features, 1024))
        vgg_model.classifier = torch.nn.Sequential(*fc_list)

        self.vgg = vgg_model

    def forward(self, x):
        deep_feats = self.top_layers(x[0])

        trfm_branch = self.pool_trfm_branch(deep_feats)

        deep_feats = torch.concat([self.denseblock4(deep_feats), trfm_branch], axis=1)

        deep_feats = self.fusion_conv(deep_feats)

        deep_feats = F.relu(deep_feats, inplace=True)

        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.size()[-1], stride=1)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # print('before concatenating')
        # print('deep feats:', deep_feats.size())
        # print('extra feats:', extra_feats.size())

        topo_mask_data = x[2]

        print('topo mask data shape:', topo_mask_data.size())
        vgg_feats = self.vgg(topo_mask_data)
        stacked_feats = torch.stack((deep_feats, extra_feats, vgg_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats, vgg_feats = trfmed_feats.unbind(0)

        # print(extra_feats.size())
        merged_feats = torch.cat([deep_feats, extra_feats, vgg_feats], axis=1)
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)
        return fc4


# 金子塔缩放模块
class PRM2(nn.Module):
    def __init__(self, img_size=224, kernel_size=4, downsample_ratio=4, dilations=[1,6,12], in_chans=3, embed_dim=64, share_weights=False, op='cat'):
        super().__init__()
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio # 这里设置的下采样率和stride的长度是一样的
        self.share_weights = share_weights # 是否共享权重，与什么模块进行权重共享呢？
        self.outSize = img_size // downsample_ratio # 计算下采样之后的图片尺寸

        if share_weights:
            self.convolution = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                stride=self.stride, padding=3*dilations[0]//2, dilation=dilations[0])

        else:
            self.convs = nn.ModuleList()
            for dilation in self.dilations:
                # padding = math.ceil(((self.kernel_size-1)*dilation + 1 - self.stride) / 2)
                padding = math.ceil(((self.kernel_size-1)*(dilation-1)) / 2)
                self.convs.append(nn.Sequential(*[nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                    stride=self.stride, padding=padding, dilation=dilation),
                    nn.GELU()]))

        if self.op == 'sum':
            self.out_chans = embed_dim
        elif op == 'cat':
            self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        B, C, W, H = x.shape
        if self.share_weights:
            padding = math.ceil(((self.kernel_size-1)*self.dilations[0] + 1 - self.stride) / 2)
            y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                stride=self.downsample_ratio, padding=padding, dilation=self.dilations[0]).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                padding = math.ceil(((self.kernel_size-1)*self.dilations[i] + 1 - self.stride) / 2)
                _y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                    stride=self.downsample_ratio, padding=padding, dilation=self.dilations[i]).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        else:
            y = self.convs[0](x).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                _y = self.convs[i](x).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        B, C, W, H, N = y.shape
        if self.op == 'sum':
            y = y.sum(dim=-1).flatten(2).permute(0,2,1).contiguous()
        elif self.op == 'cat':
            y = y.permute(0,4,1,2,3).flatten(3).reshape(B, N*C, W, H)
        else:
            raise NotImplementedError('no such operation: {} for multi-levels!'.format(self.op))
        return y


# 把densenet121改变成resnet50, 并增加trfm;增加PRM2模块
class Net23(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net23, self).__init__()
        self.PRM2 = PRM2()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.PRM2(x[0])
        deep_feats = self.res_layer(deep_feats)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 把densenet121改变成resnet50, 并增加trfm;增加PRM2模块
class Net24(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net24, self).__init__()
        self.prm = PRM2(img_size=256, kernel_size=4, downsample_ratio=4, dilations=[1,6,12], in_chans=3, embed_dim=16)
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.prm(x[0])
        deep_feats = self.res_layer(deep_feats)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 把densenet121改变成resnet50, 并替换trfm为fc融合层;增加PRM模块
class Net25(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net25, self).__init__()
        # self.prm = PRM2(img_size=256, kernel_size=4, downsample_ratio=4, dilations=[1,6,12], in_chans=3, embed_dim=16)
        # self.prm = PRM2(img_size=256, kernel_size=4, downsample_ratio=4, dilations=[1, 6, 12, 18, 24], in_chans=3, embed_dim=16)
        # self.prm = PRM2(img_size=256, kernel_size=4, downsample_ratio=4, dilations=[1, 6, 12], in_chans=3, embed_dim=8)
        # self.prm = PRM2(img_size=256, kernel_size=4, downsample_ratio=4, dilations=[1, 6, 12, 18, 24], in_chans=3, embed_dim=32)

        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)

    def forward(self, x):
        deep_feats = self.prm(x[0])
        deep_feats = self.res_layer(deep_feats)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 把densenet121改变成resnet50, 并替换trfm为fc融合层;增加2个PRM模块
class Net27(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net27, self).__init__()
        self.prm1 = PRM2(img_size=256, kernel_size=4, downsample_ratio=4, dilations=[1,6,12], in_chans=3, embed_dim=16)
        self.prm2 = PRM2(img_size=64, kernel_size=4, downsample_ratio=4, dilations=[1, 6, 12], in_chans=3 * 16, embed_dim=16)
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)

    def forward(self, x):
        deep_feats = self.prm1(x[0])
        deep_feats = self.prm2(deep_feats)
        deep_feats = self.res_layer(deep_feats)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2



class Net26(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net26, self).__init__()
        self.prm = PRM2(img_size=256, kernel_size=4, downsample_ratio=4, dilations=[1,3,6,12], in_chans=3, embed_dim=64)
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)

    def forward(self, x):
        deep_feats = self.prm(x[0])
        deep_feats = self.res_layer(deep_feats)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net28(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net28, self).__init__()
        # self.stem = ReductionCell(in_chans=3, embed_dims=64, token_dims=64)
        # self.stem = ReductionCell(in_chans=3, embed_dims=16, token_dims=16)
        # self.stem = ReductionCell(in_chans=3, embed_dims=8, token_dims=8)
        # self.stem = ReductionCell(in_chans=3, embed_dims=16, token_dims=16, tokens_type = 'pooling')
        # self.stem = ReductionCell(in_chans=3, embed_dims=16, token_dims=16, downsample_ratios=8, tokens_type = 'pooling')
        # self.stem = ReductionCell(in_chans=3, embed_dims=3, token_dims=3)
        # self.stem = ReductionCell(in_chans=3, embed_dims=5, token_dims=5)

        # self.stem = ReductionCell(in_chans=3, embed_dims=3, token_dims=3)
        # self.stem2 = ReductionCell(in_chans=3, embed_dims=3, token_dims=3)

        self.stem = ReductionCell(in_chans=3, embed_dims=10, token_dims=10)

        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        # self.dp = nn.Dropout(0.2)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.stem(x[0])
        # deep_feats = self.stem2(deep_feats)
        deep_feats = self.res_layer(deep_feats)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        # fc2 = self.dp(fc2)
        return fc2



class Net29(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net29, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net30(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net30, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.dp = nn.Dropout(p=0.2, inplace=False)
        self.threshold = torch.nn.Parameter(torch.as_tensor(-1.0).double())
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        input = torch.where(x[0].double() <= self.threshold, 0.0, x[0].double())
        input = input.float()
        deep_feats = self.res_layer(input)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        fc2 = self.dp(fc2)
        return fc2


class Net31(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net31, self).__init__()
        self.res_layer = model
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.dp = nn.Dropout(p=0.2, inplace=False)
        self.attention_mask = torch.nn.Parameter(torch.ones((256, 256)).float())
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        input = self.attention_mask * x[0]
        deep_feats = self.res_layer(input)
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        fc2 = self.dp(fc2)
        return fc2


class Net32(nn.Module):
    def __init__(self, model, num_classes=2):
        super(Net32, self).__init__()
        self.res_layer = model
        self.classifier_fc = nn.Linear(1000, num_classes, bias=True)
    
    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        fc = self.classifier_fc(deep_feats)

        return fc



class Net32Plus(nn.Module):
    def __init__(self):
        super(Net32Plus, self).__init__()

        res = models.resnet50(pretrained=True)
        self.res_layer = nn.Sequential(*list(res.children())[:-1])
        self.fc = nn.Linear(2048, 2, bias=True)


    def forward(self, x):
        x = self.res_layer(x[0])
        # print("x shape:", x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Net32Plus_W_DeepLatent(nn.Module):
    def __init__(self, num_classes = 2, latent_from_dim = 10, latent_to_dim = 10):
        super(Net32Plus_W_DeepLatent, self).__init__()

        res = models.resnet50(pretrained=True)
        self.res_layer = nn.Sequential(*list(res.children())[:-1])
        self.deep_latent = torch.nn.Parameter(torch.randn(latent_from_dim), requires_grad=True)
        self.latent_fc = nn.Linear(latent_from_dim, latent_to_dim)
        self.fc = nn.Linear(2048 + latent_to_dim, num_classes, bias=True)


    def forward(self, x):
        x = self.res_layer(x[0])
        # print("x shape:", x.shape)
        x = x.view(x.shape[0], -1)

        deep_latent = self.latent_fc(self.deep_latent)
        deep_latent = deep_latent.repeat(x.shape[0], 1)
        x = torch.cat((x, deep_latent), -1)
        x = self.fc(x)
        return x



class Net32Plus_W_DeepLatent2(nn.Module):
    def __init__(self, num_classes = 2,  latent_to_dim = 10):
        super(Net32Plus_W_DeepLatent2, self).__init__()

        res = models.resnet50(pretrained=True)
        self.res_layer = nn.Sequential(*list(res.children())[:-1])
        self.deep_latent = torch.nn.Parameter(torch.randn(latent_to_dim), requires_grad=True)
        self.fc = nn.Linear(2048 + latent_to_dim, num_classes, bias=True)


    def forward(self, x):
        x = self.res_layer(x[0])
        # print("x shape:", x.shape)
        x = x.view(x.shape[0], -1)

        deep_latent = self.deep_latent.repeat(x.shape[0], 1)
        x = torch.cat((x, deep_latent), -1)
        x = self.fc(x)
        return x



class Net33(nn.Module):
    def __init__(self, model, dense_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net33, self).__init__()
        self.res_layer = model
        self.dense_layer = dense_model.res_layer
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + 1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        deep_feats_2 = self.dense_layer(x[0])
        deep_feats_2 = deep_feats_2.view(deep_feats_2.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats, deep_feats_2), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats, deep_feats_2 = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats, deep_feats_2], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net34(nn.Module):
    def __init__(self, model, dense_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net34, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-2])
        self.dense_layer = nn.Sequential(*list(dense_model.res_layer.children())[:-1])
        # self.conv_fusion = nn.Conv2d(2048 + 1024, 1000, kernel_size=(1,1), stride=(1,1), bias=False)
        # self.bn_fusion = nn.BatchNorm2d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.relu_fusion = nn.ReLU(inplace=True)

        # self.fusion_fc = nn.Linear(2048, 1000, bias=True)
        self.fusion_fc = nn.Linear(3072, 1000, bias=True)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats_2 = self.dense_layer(x[0])

        fusion_deep_feats = torch.cat((deep_feats, deep_feats_2), 1)
        # fusion_deep_feats = self.conv_fusion(fusion_deep_feats)
        # fusion_deep_feats = self.bn_fusion(fusion_deep_feats)
        # fusion_deep_feats = self.relu_fusion(fusion_deep_feats)

        fusion_deep_feats = F.avg_pool2d(fusion_deep_feats, kernel_size=fusion_deep_feats.shape[-1])
        fusion_deep_feats = fusion_deep_feats.view(fusion_deep_feats.shape[0], -1)
        fusion_deep_feats = self.fusion_fc(fusion_deep_feats)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((fusion_deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        fusion_deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([fusion_deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2



class Net35(nn.Module):
    def __init__(self, model, dense_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net35, self).__init__()
        self.res_layer = model
        self.dense_layer = dense_model.res_layer

        # self.fusion_fc = nn.Linear(2048, 1000, bias=True)
        self.fusion_fc = nn.Linear(3072, 1000, bias=True)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats_2 = self.dense_layer(x[0])
        
        fusion_deep_feats = torch.add(deep_feats, deep_feats_2) / 2.0

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((fusion_deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        fusion_deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([fusion_deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2



class Net35_W_DeepLatent(nn.Module):
    def __init__(self, model, dense_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2, latent_from_dim=10, latent_to_dim=100):
        super(Net35_W_DeepLatent, self).__init__()
        self.res_layer = model
        self.dense_layer = dense_model.res_layer

        # self.fusion_fc = nn.Linear(2048, 1000, bias=True)
        self.fusion_fc = nn.Linear(3072, 1000, bias=True)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)

        self.deep_latent = torch.nn.Parameter(torch.randn(latent_from_dim), requires_grad=True)
        self.deep_latent_fc = nn.Linear(latent_from_dim, latent_to_dim)
        self.fc2 = nn.Linear(1000 + num_mid_feats + latent_to_dim, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)


    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats_2 = self.dense_layer(x[0])
        
        fusion_deep_feats = torch.add(deep_feats, deep_feats_2) / 2.0

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((fusion_deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        fusion_deep_feats, extra_feats = trfmed_feats.unbind(0)

        deep_latent_feats = self.deep_latent_fc(self.deep_latent)
        deep_latent_feats = deep_latent_feats.repeat(extra_feats.shape[0], 1)

        merged_feats = torch.cat([fusion_deep_feats, extra_feats, deep_latent_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2



class Net35Pretrained(nn.Module):
    def __init__(self, model, dense_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net35Pretrained, self).__init__()
        self.res_layer = model.model # model是netwrapper类型的
        self.dense_layer = dense_model.res_layer

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats_2 = self.dense_layer(x[0])
        
        fusion_deep_feats = torch.add(deep_feats, deep_feats_2) / 2.0

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((fusion_deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        fusion_deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([fusion_deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


# 冻结参数不训练
class Net35_2(nn.Module):
    def __init__(self, model, dense_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net35_2, self).__init__()
        self.res_layer = model
        self.dense_layer = dense_model.res_layer

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

        for param in self.dense_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats_2 = self.dense_layer(x[0])
        
        fusion_deep_feats = torch.add(deep_feats, deep_feats_2) / 2.0

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((fusion_deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        fusion_deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([fusion_deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2



# 冻结参数不训练并增加参数转化层
class Net35_3(nn.Module):
    def __init__(self, model, dense_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net35_3, self).__init__()
        self.res_layer = model
        self.dense_layer = dense_model.res_layer
        self.dense_layer_fc = nn.Linear(1000, 1000)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

        for param in self.dense_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats_2 = self.dense_layer(x[0])
        deep_feats_2 = self.dense_layer_fc(deep_feats_2)
        
        fusion_deep_feats = torch.add(deep_feats, deep_feats_2) / 2.0

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((fusion_deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        fusion_deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([fusion_deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2



class Net35_4(nn.Module):
    def __init__(self, model, dense_model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net35_4, self).__init__()
        self.res_layer = model
        self.dense_layer = dense_model.res_layer

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

        for param in self.dense_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats_2 = self.dense_layer(x[0])
        deep_feats_2 = self.dense_layer_fc(deep_feats_2)
        
        fusion_deep_feats = torch.add(deep_feats, deep_feats_2) / 2.0

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((fusion_deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        fusion_deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([fusion_deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class Net36(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net36, self).__init__()
        self.res_layer = nn.Sequential(*list(model.children())[:-2])
        self.res_layer_fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)
        self.attention_mask = torch.nn.Parameter(torch.ones((1, 1, 8, 8)).float(), requires_grad=True)
        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        # print(self.attention_mask.repeat(deep_feats.shape[0], deep_feats.shape[1], 1,1).shape)

        deep_feats =  deep_feats * self.attention_mask.repeat(deep_feats.shape[0], deep_feats.shape[1], 1,1)

        deep_feats = F.avg_pool2d(deep_feats, kernel_size=deep_feats.shape[-1])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        deep_feats = self.res_layer_fc(deep_feats)

        # print(deep_feats.size())

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        # print(merged_feats.size())

        fc2 = self.fc2(merged_feats)
        return fc2


class NetWrapper(nn.Module):
    def __init__(self, model, num_classes=2):
        super(NetWrapper, self).__init__()
        self.model = model
        self.fc = nn.Linear(1000, num_classes, bias=True)
      
    def forward(self, x):
        output = self.model(x[0])
        output = self.fc(output)
        return output


class NetMerge(nn.Module):
    def __init__(self, num_classes = 2, has_second_fc = True):
        super().__init__()
        densenet = models.densenet121(pretrained=True)
        resnet = models.resnet50(pretrained=True)
        self.densenet_body = nn.Sequential(*list(densenet.children())[0][4:5])
        self.resnet_head = nn.Sequential(*list(resnet.children())[:4])
        self.resnet_body = nn.Sequential(*list(resnet.children())[4])
        self.resnet_tail = nn.Sequential(*list(resnet.children())[5:9])
        self.fc1 = nn.Linear(2048, 1000, bias=True)

        self.has_second_fc = has_second_fc
        if has_second_fc:
            self.fc2 = nn.Linear(1000, num_classes, bias=True)
        

    def forward(self, x):
        if type(x) == tuple:
            head_output = self.resnet_head(x[0])
        else:
            head_output = self.resnet_head(x)
        resnet_body_output = self.resnet_body(head_output)
        densenet_body_output = self.densenet_body(head_output)
        
        x = (resnet_body_output + densenet_body_output)/2
        x = self.resnet_tail(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        if self.has_second_fc:
            x = self.fc2(x)

        return x



class NetMergeWExtra(nn.Module):
    def __init__(self, backbone, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(NetMergeWExtra, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)

        self.fc2 = nn.Linear(1000 + num_mid_feats, num_classes)

        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)


    def forward(self, x):
        deep_feats = self.backbone(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        fc2 = self.fc2(merged_feats)
        return fc2


class NetMergeWExtraWLatent(nn.Module):
    def __init__(self, backbone, num_extra_feats=25, num_mid_feats=1000, latent_from_dim=10, latent_to_dim=100, num_classes=2):
        super(NetMergeWExtraWLatent, self).__init__()
        self.backbone = backbone

        self.deep_latent = torch.nn.Parameter(torch.randn(latent_from_dim), requires_grad=True)
        self.latent_fc = nn.Linear(latent_from_dim, latent_to_dim)
 
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)

        self.fc2 = nn.Linear(1000 + num_mid_feats + latent_to_dim, num_classes)

        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)


    def forward(self, x):
        deep_feats = self.backbone(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)

        deep_latent = self.latent_fc(self.deep_latent).repeat(deep_feats.shape[0], 1)
        merged_feats = torch.cat([deep_feats, extra_feats, deep_latent], axis=1)
      
        fc2 = self.fc2(merged_feats)
        return fc2


class DenseNet1(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet1, self).__init__()
        dense = models.densenet121(pretrained=True)
        self.model = nn.Sequential(*list(dense.children())[0])
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = self.avg_pool(res)
        res = res.view(res.shape[0], -1)
        res = self.fc(res)
        return res


class ResNet1(nn.Module):
    def __init__(self, num_classes = 2):
        super(ResNet1, self).__init__()       
        res = models.resnet50(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:7])
        self.avg_pool = res_list[8]
        self.fc = nn.Linear(1024,num_classes)

    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = self.avg_pool(res)
        res = res.view(res.shape[0], -1)
        res = self.fc(res)
        return res


class ResNet2(nn.Module):
    def __init__(self, num_classes = 2):
        super(ResNet2, self).__init__()       
        res = models.resnet50(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:7])
        self.avg_pool = res_list[8]
        self.fc = nn.Linear(1024,500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = self.avg_pool(res)
        res = res.view(res.shape[0], -1)
        res = self.fc(res)
        res = self.fc2(res)
        return res


class ResNet3(nn.Module):
    def __init__(self, num_classes = 2):
        super(ResNet3, self).__init__()       
        res = models.resnet50(pretrained=True)
        self.model = res
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = self.fc(res)
        return res
        

# 
class ResNet4(nn.Module):
    def __init__(self, num_classes = 2):
        super(ResNet4, self).__init__() 
        res = models.resnet34(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:-1])
        self.fc = nn.Linear(res_list[-1].in_features,num_classes)

    def forward(self, x):   
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = res.view(res.shape[0], -1)
        res = self.fc(res)
        return res
        
    
class ResNet5(nn.Module):
    def __init__(self, num_classes = 2):
        super(ResNet5, self).__init__() 
        res = models.resnet18(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:-1])
        self.fc = nn.Linear(res_list[-1].in_features,num_classes)

        for m in self.named_modules():
            if 'model' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                print("module:", m)
                print("---------------------------------")
                pass


    def forward(self, x):   
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = res.view(res.shape[0], -1)
        res = self.fc(res)
        return res




class ResNet6(nn.Module):
    def __init__(self, num_classes = 2):
        super(ResNet6, self).__init__() 
        res = models.resnet101(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:-1])
        self.fc = nn.Linear(res_list[-1].in_features,num_classes)

    def forward(self, x):   
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = res.view(res.shape[0], -1)
        res = self.fc(res)
        return res


class ResNet7(nn.Module):
    def __init__(self, mid_classes = 100, num_classes = 2) -> None:
        super().__init__()
        res = models.resnet50(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:-1])
        self.fc1 = nn.Linear(res_list[-1].in_features, mid_classes)
        self.fc2 = nn.Linear(mid_classes, num_classes)
        
    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = res.view(res.shape[0], -1)
        res = self.fc1(res)
        res = self.fc2(res)
        return res


class ResNet8(nn.Module):
    def __init__(self):
        super(ResNet8, self).__init__()

        res = models.resnet50(pretrained=True)
        res_list = list(res.children())
        self.before_res_blocks = nn.Sequential(*res_list[:-3])
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.last_res_blocks = nn.Sequential(*res_list[-3:-1])
        self.fc = nn.Linear(res_list[-1].in_features + res_list[-4][-1].conv3.out_channels, 2, bias=True) # (1024 + 2-48, 2)

    def forward(self, x):
        if type(x) == tuple:
            output = self.before_res_blocks(x[0])
        else:
            output = self.before_res_blocks(x)
        output = self.pool1(output)
        output_v =  output.view(output.shape[0], -1)

        output2 = self.last_res_blocks(output)
        output2_v = output2.view(output2.shape[0], -1)
        output_concat = torch.concat((output_v, output2_v), -1)

        output_concat = self.fc(output_concat)
        return output_concat



class ResNet9(nn.Module):
    def __init__(self, num_extra_feats=10, num_mid_feats=50, num_classes=2):
        super(ResNet9, self).__init__()
        res = models.resnet50(pretrained=True)
        res_list = list(res.children())
        self.res_layer = nn.Sequential(*res_list[:-1])
        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(2048 + num_mid_feats, num_classes)


    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)
      
        fc2 = self.fc2(merged_feats)
        return fc2




# 在resnet18的基础上加上注意力融合模块
class ResNet10(nn.Module):
    def __init__(self, num_extra_feats=10, num_mid_feats=128, num_classes=2):
        super(ResNet10, self).__init__()

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:-1])
        self.modified_fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=True)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
       
        self.fc2 = nn.Linear(num_mid_feats * 2, num_classes)

        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

        for m in self.named_modules():
            if 'model' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        deep_feats = self.modified_fc(deep_feats)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)

        fc2 = self.fc2(merged_feats)

        return fc2




# 在resnet18的基础上加上注意力融合模块
class ResNet11(nn.Module):
    def __init__(self, num_extra_feats=10, num_mid_feats=32, num_classes=2):
        super(ResNet11, self).__init__()

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:-1])
        self.modified_fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=True)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
       
        self.fc2 = nn.Linear(num_mid_feats , num_classes)

        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

        for m in self.named_modules():
            if 'model' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        deep_feats = self.modified_fc(deep_feats)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats, extra_feats = trfmed_feats.unbind(0)
        merged_feats = deep_feats

        fc2 = self.fc2(merged_feats)

        return fc2



class ResNet12(nn.Module):
    def __init__(self, num_extra_feats=10, num_mid_feats=128, num_classes=2):
        super(ResNet12, self).__init__()

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())
        self.model = nn.Sequential(*res_list[:-1])
        self.modified_fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=True)

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
       
        self.fc2 = nn.Linear(num_mid_feats * 2, num_classes)


        for m in self.named_modules():
            if 'model' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        deep_feats = self.modified_fc(deep_feats)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        merged_feats = torch.cat([deep_feats, extra_feats], axis=1)

        fc2 = self.fc2(merged_feats)

        return fc2




class ResNet13(nn.Module):
    def __init__(self,  num_mid_feats=32, num_classes=2):
        super(ResNet13, self).__init__()

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())
        res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats)
        self.model = res
       
        self.fc = nn.Linear(num_mid_feats , num_classes)

        for m in self.named_modules():
            if 'model' in m[0] and 'model.fc' not in m[0]:
                continue
            if '' == m[0]:
                continue
            

            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        fc = self.fc(deep_feats)

        return fc


class WeightedPool(nn.Module):

    def __init__(self, channels_dim = 1024):

        super(WeightedPool, self).__init__()
        self.weight_latent = torch.nn.Parameter(torch.ones(channels_dim) * 0.5, requires_grad=True)


    def forward(self, x):
        
        res = x.view(x.shape[0], x.shape[1], -1)
        res = res.permute(0,2,1)

        weight_latent = self.weight_latent.unsqueeze(-1)
        
        weights = torch.matmul(res, weight_latent)

        weights = weights.squeeze(-1)

        weights = weights.unsqueeze(1)

        weights = F.softmax(weights, -1)

        updated_res = torch.matmul(weights, res)
        updated_res = updated_res.squeeze(1)

        return updated_res



class WeightedPool2(nn.Module):

    def __init__(self, num_trans_dim = 5):

        super(WeightedPool2, self).__init__()
        self.weight_latent = torch.nn.Parameter(torch.rand(num_trans_dim, 1), requires_grad=True)
        self.feat_dim = nn.Linear(1, num_trans_dim, bias=True)

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        res = x.view(x.shape[0], x.shape[1], -1)
        res = res.permute(0,2,1)

        res_expand = res.unsqueeze(-1)
        
        res_expand = self.feat_dim(res_expand)

        weight = torch.matmul(res_expand, self.weight_latent)
        weight = weight.squeeze(-1)
        weight = F.softmax(weight, dim=1)

        weight_sum = weight * res
        weight_sum = torch.sum(weight_sum, 1)

        return weight_sum
        


class WeightedPool3(nn.Module):

    def __init__(self, num_trans_dim = 5):

        super(WeightedPool3, self).__init__()
        self.weight_latent_1 = torch.nn.Parameter(torch.rand(num_trans_dim, 1), requires_grad=True)
        self.weight_latent_2 = torch.nn.Parameter(torch.rand(num_trans_dim, 1), requires_grad=True)
        self.feat_dim_1 = nn.Linear(1, num_trans_dim, bias=True)
        self.feat_dim_2 = nn.Linear(1, num_trans_dim, bias=True)

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        res = x.view(x.shape[0], x.shape[1], -1)
        res = res.permute(0,2,1)

        res_expand = res.unsqueeze(-1)
        
        res_expand_1 = self.feat_dim_1(res_expand)

        weight_1= torch.matmul(res_expand_1, self.weight_latent_1)
        weight_1 = weight_1.squeeze(-1)
        weight_1 = F.softmax(weight_1, dim=1)

        weight_sum_1 = weight_1 * res
        weight_sum_1 = torch.sum(weight_sum_1, 1)

        res_expand_2 = self.feat_dim_2(res_expand)

        weight_2= torch.matmul(res_expand_2, self.weight_latent_2)
        weight_2 = weight_2.squeeze(-1)
        weight_2 = F.softmax(weight_2, dim=1)

        weight_sum_2 = weight_2 * res
        weight_sum_2 = torch.sum(weight_sum_2, 1)

        # weight_sum = torch.cat((weight_sum_1, weight_sum_2), -1)

        weight_sum = (weight_sum_1 + weight_sum_2) / 2

        return weight_sum





class WeightedPool4(nn.Module):

    def __init__(self, num_trans_dim = 5):

        super(WeightedPool4, self).__init__()
        self.weight_latent_1 = torch.nn.Parameter(torch.rand(num_trans_dim, 1), requires_grad=True)
        self.weight_latent_2 = torch.nn.Parameter(torch.rand(num_trans_dim, 1), requires_grad=True)
        self.weight_latent_3 = torch.nn.Parameter(torch.rand(num_trans_dim, 1), requires_grad=True)
        self.feat_dim_1 = nn.Linear(1, num_trans_dim, bias=True)
        self.feat_dim_2 = nn.Linear(1, num_trans_dim, bias=True)
        self.feat_dim_3 = nn.Linear(1, num_trans_dim, bias=True)

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        res = x.view(x.shape[0], x.shape[1], -1)
        res = res.permute(0,2,1)

        res_expand = res.unsqueeze(-1)
        
        res_expand_1 = self.feat_dim_1(res_expand)

        weight_1= torch.matmul(res_expand_1, self.weight_latent_1)
        weight_1 = weight_1.squeeze(-1)
        weight_1 = F.softmax(weight_1, dim=1)

        weight_sum_1 = weight_1 * res
        weight_sum_1 = torch.sum(weight_sum_1, 1)

        res_expand_2 = self.feat_dim_2(res_expand)

        weight_2= torch.matmul(res_expand_2, self.weight_latent_2)
        weight_2 = weight_2.squeeze(-1)
        weight_2 = F.softmax(weight_2, dim=1)

        weight_sum_2 = weight_2 * res
        weight_sum_2 = torch.sum(weight_sum_2, 1)


        res_expand_3 = self.feat_dim_3(res_expand)
        weight_3 = torch.matmul(res_expand_3, self.weight_latent_3)
        weight_3 = weight_3.squeeze(-1)
        weight_3 = F.softmax(weight_3, dim=1)

        weight_sum_3 = weight_3 * res
        weight_sum_3 = torch.sum(weight_sum_3, 1)

        # weight_sum = torch.cat((weight_sum_1, weight_sum_2), -1)

        weight_sum = (weight_sum_1 + weight_sum_2 + weight_sum_3) / 2

        return weight_sum



class ResNet13_Change_Pool(nn.Module):
    def __init__(self,  num_mid_feats=32, num_classes=2):
        super(ResNet13_Change_Pool, self).__init__()

        # res = models.resnet18(pretrained=True)
        # res_list = list(res.children())
        # res.avgpool = WeightedPool(channels_dim=512)
        # # res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats)
        # res.fc = nn.Linear(res_list[-1].in_features, num_classes)

        # self.model = res

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())

        new_res_list = res_list[:-2]
        new_res_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        new_res_list.append(WeightedPool(512))
        new_res_list.append(nn.Linear(res_list[-1].in_features, num_mid_feats))

        self.model = nn.Sequential(*new_res_list)
       
        self.fc = nn.Linear(num_mid_feats , num_classes)

        for m in self.named_modules():
            if 'model' in m[0] and 'model.fc' not in m[0]:
                continue
            if '' == m[0]:
                continue
            

            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        fc = self.fc(deep_feats)
        return fc

        # return deep_feats



class ResNet13_Change_Pool2(nn.Module):
    def __init__(self,  num_mid_feats=32, num_classes=2):
        super(ResNet13_Change_Pool2, self).__init__()

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())
        res.avgpool = WeightedPool2()

        res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats)
        # res.fc = nn.Linear(res_list[-1].in_features, num_classes)

        self.model = res
       
        self.fc = nn.Linear(num_mid_feats , num_classes)

        for m in self.named_modules():
            if 'model' in m[0] and 'model.fc' not in m[0]:
                continue
            if '' == m[0]:
                continue
            

            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        fc = self.fc(deep_feats)
        return fc

        # return deep_feats





class ResNet13_Change_Pool3(nn.Module):
    def __init__(self,  num_mid_feats=32, num_classes=2):
        super(ResNet13_Change_Pool3, self).__init__()

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())

        new_res_list = res_list[:-2]
        # new_res_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # new_res_list.append(nn.AdaptiveAvgPool2d(output_size=(2, 2)))
        new_res_list.append(WeightedPool2(num_trans_dim=32))
        new_res_list.append(nn.Linear(res_list[-1].in_features, num_mid_feats))

        self.model = nn.Sequential(*new_res_list)
       
        self.fc = nn.Linear(num_mid_feats , num_classes)

        for m in self.named_modules():
            if 'model' in m[0] and 'model.fc' not in m[0]:
                continue
            if '' == m[0]:
                continue
            

            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        fc = self.fc(deep_feats)
        return fc

        # return deep_feats


class ResNet13_Change_Pool3_2(nn.Module):
    def __init__(self,  num_mid_feats=32, num_classes=2):
        super(ResNet13_Change_Pool3_2, self).__init__()

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())

        new_res_list = res_list[:-2]
        # new_res_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        new_res_list.append(nn.AdaptiveAvgPool2d(output_size=(2, 2)))
        new_res_list.append(WeightedPool2(num_trans_dim=30))
        new_res_list.append(nn.Linear(res_list[-1].in_features, num_mid_feats))

        self.model = nn.Sequential(*new_res_list)
       
        self.fc = nn.Linear(num_mid_feats , num_classes)

        for m in self.named_modules():
            if 'model' in m[0] and 'model.fc' not in m[0]:
                continue
            if '' == m[0]:
                continue

            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass

    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        fc = self.fc(deep_feats)
        return fc

        # return deep_feats



class ResNet13_Change_Pool4(nn.Module):
    def __init__(self,  num_mid_feats=32, num_classes=2):
        super(ResNet13_Change_Pool4, self).__init__()

        res = models.resnet18(pretrained=True)
        res_list = list(res.children())

        new_res_list = res_list[:-2]
        # new_res_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # new_res_list.append(nn.AdaptiveAvgPool2d(output_size=(2, 2)))
        new_res_list.append(WeightedPool3(num_trans_dim=35))
        new_res_list.append(nn.Linear(res_list[-1].in_features * 2, num_mid_feats))

        self.model = nn.Sequential(*new_res_list)
       
        self.fc = nn.Linear(num_mid_feats , num_classes)

        for m in self.named_modules():
            if 'model' in m[0] and 'model.fc' not in m[0]:
                continue
            if '' == m[0]:
                continue
            

            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        fc = self.fc(deep_feats)
        return fc

        # return deep_feats



class ResNet14(nn.Module):
    def __init__(self, model, num_extra_feats=10, num_mid_feats=32, num_classes=2):
        super(ResNet14, self).__init__()

        self.model = model.model

        self.fc1 = nn.Linear(num_extra_feats, num_mid_feats)
       
        # self.fc2 = nn.Linear(num_mid_feats * 2 , num_classes)
        self.fc2 = nn.Linear(num_mid_feats  , num_classes)

        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)

        stacked_feats = torch.stack((deep_feats, extra_feats), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats0, extra_feats0 = trfmed_feats.unbind(0)
        # deep_feats = (deep_feats + deep_feats0) 

        deep_feats = deep_feats + deep_feats0
        # extra_feats = extra_feats0 + extra_feats
        # extra_feats = extra_feats0

        # merged_feats = torch.cat((deep_feats, extra_feats), -1)
        merged_feats = deep_feats

        fc2 = self.fc2(merged_feats)

        return fc2



class ResNet15(nn.Module):
    def __init__(self, model, num_extra_feats=10, num_mid_feats=32, num_classes=2):
        super(ResNet15, self).__init__()

        self.model = model.model
       
        self.fc = nn.Linear(num_mid_feats + num_extra_feats , num_classes)


        for m in self.named_modules():
            if 'model' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]

        merged_feats = torch.cat((deep_feats, extra_feats), -1)

        res = self.fc(merged_feats)

        return res


class ResNet16(nn.Module):
    def __init__(self, num_extra_features = 10, num_mid_features = 32, num_classes = 2):
        super().__init__()
        self.fc1 = nn.Linear(num_extra_features, num_mid_features)
        self.fc2 = nn.Linear(num_mid_features, num_classes)
        # self.relu = nn.ReLU(inplace=False)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        # extra_feats = self.relu(extra_feats)
        extra_feats = self.fc2(extra_feats)

        return extra_feats
        



class ResNet17(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2):
        super(ResNet17, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model
       
        self.fc = nn.Linear(num_mid_feats , num_classes)

        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

        for param in self.model1.parameters():
            param.requires_grad = False

        for param in self.model2.parameters():
            param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        stacked_feats = torch.stack((deep_feats1, deep_feats2), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats1_update, deep_feats2_update = trfmed_feats.unbind(0)

        # deep_feats = deep_feats1_update + deep_feats2_update
        
        deep_feats = deep_feats1_update + deep_feats1
    
        merged_feats = deep_feats

        fc = self.fc(merged_feats)

        return fc



class ResNet17_2(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2):
        super(ResNet17_2, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model
       
        # self.fc = nn.Linear(num_mid_feats * 2 , num_classes)
        self.fc = nn.Linear(num_mid_feats , num_classes)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        merged_feats = (deep_feats2 + deep_feats1) / 2
    
        # merged_feats = torch.cat((deep_feats1, deep_feats2), -1)

        fc = self.fc(merged_feats)

        return fc



class ResNet17_3(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2):
        super(ResNet17_3, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model
       
        # self.fc = nn.Linear(num_mid_feats * 2 , num_classes)
        # self.fc = nn.Linear(num_mid_feats + 5, num_classes)
        self.fc = nn.Linear(num_mid_feats, num_classes)

        # self.weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        # self.weight = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.weight1 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # self.fc1 = nn.Linear(num_mid_feats, num_mid_feats)
        # self.fc2 = nn.Linear(num_mid_feats, num_mid_feats)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        # deep_feats1 = self.fc1(deep_feats1)
        # deep_feats2 = self.fc2(deep_feats2)

        # merged_feats = (deep_feats1 + deep_feats2) / 2
        # merged_feats = deep_feats1 * 0.2 + deep_feats2 * 0.8

        # merged_feats = deep_feats1 * self.weight + deep_feats2 * (1 - self.weight)
                
        merged_feats = deep_feats1 * self.weight1 + deep_feats2 * self.weight2
    
        # merged_feats = torch.cat((deep_feats1, deep_feats2), -1)

        # rand_v = self.random_v.repeat(merged_feats.shape[0], 1)
        # merged_feats = torch.cat((merged_feats, rand_v), -1)

        fc = self.fc(merged_feats)

        return fc



class ResNet17_3(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2):
        super(ResNet17_3, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model
       
        # self.fc = nn.Linear(num_mid_feats * 2 , num_classes)
        # self.fc = nn.Linear(num_mid_feats + 5, num_classes)
        self.fc = nn.Linear(num_mid_feats, num_classes)

        # self.weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        # self.weight = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.weight1 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # self.fc1 = nn.Linear(num_mid_feats, num_mid_feats)
        # self.fc2 = nn.Linear(num_mid_feats, num_mid_feats)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        # deep_feats1 = self.fc1(deep_feats1)
        # deep_feats2 = self.fc2(deep_feats2)

        # merged_feats = (deep_feats1 + deep_feats2) / 2
        # merged_feats = deep_feats1 * 0.2 + deep_feats2 * 0.8

        # merged_feats = deep_feats1 * self.weight + deep_feats2 * (1 - self.weight)
                
        merged_feats = deep_feats1 * self.weight1 + deep_feats2 * self.weight2
    
        # merged_feats = torch.cat((deep_feats1, deep_feats2), -1)

        # rand_v = self.random_v.repeat(merged_feats.shape[0], 1)
        # merged_feats = torch.cat((merged_feats, rand_v), -1)

        fc = self.fc(merged_feats)

        return fc




class ResNet17_4(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2):
        super(ResNet17_4, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.weight_latent = torch.nn.Parameter(torch.rand(num_mid_feats), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        weight_latent = self.weight_latent.view(-1, 1) # (32,1)
        feats = torch.stack((deep_feats1, deep_feats2), 1) # (5,2,32)
        
        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1 * weight_0 + deep_feats2 * weight_1

        fc = self.fc(merged_feats)

        return fc



class ResNet17_4_Loss_Changed(nn.Module):
    def __init__(self, num_mid_feats=32, num_classes=2):
        super(ResNet17_4_Loss_Changed, self).__init__()


        res = models.resnet18(pretrained=True)
        res_list = list(res.children())
        res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=False)
        self.model1 = res
        self.fc1 = nn.Linear(num_mid_feats, num_classes)


        dense = models.densenet121(pretrained=True)
        dense.classifier = nn.Linear(1024, num_mid_feats, bias=True)
        self.model2 = dense
        self.fc2 = nn.Linear(num_mid_feats, num_classes)


        self.weight_latent = torch.nn.Parameter(torch.rand(num_mid_feats), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)


        deep_feats1_fc = self.fc1(deep_feats1)
        deep_feats2_fc = self.fc2(deep_feats2)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        weight_latent = self.weight_latent.view(-1, 1) # (32,1)
        feats = torch.stack((deep_feats1, deep_feats2), 1) # (5,2,32)
        
        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1 * weight_0 + deep_feats2 * weight_1

        fc = self.fc(merged_feats)

        return fc, deep_feats1_fc, deep_feats2_fc



def add_key(a_ori,b):
    a = a_ori.copy()
    for key, value in b.items():
        if key in a.keys():
            continue
        else:
            a[key] = value
    return a


class ResNet17_4_Refined_W_Loss_New(nn.Module):
    
    def __init__(self, num_mid_feats=32, num_classes=2, num_trans_dim=8):
            super(ResNet17_4_Refined_W_Loss_New, self).__init__()

            
            res = models.resnet18(pretrained=True)
            res_list = list(res.children())
            res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=False)
            self.model1 = res
            self.fc1 = nn.Linear(num_mid_feats, num_classes)


            dense = models.densenet121(pretrained=True)
            dense.classifier = nn.Linear(1024, num_mid_feats, bias=True)
            self.model2 = dense
            self.fc2 = nn.Linear(num_mid_feats, num_classes)


            self.weight_latent = torch.nn.Parameter(torch.rand(size=(num_trans_dim,1)), requires_grad=True)

            self.fc = nn.Linear(num_mid_feats, num_classes)

            self.feat_dim = nn.Linear(1, num_trans_dim)


            self.res_weights_dict = {}
            self.dense_weights_dict = {}

            # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

            # for param in self.model1.parameters():
            #     param.requires_grad = False

            # for param in self.model2.parameters():
            #     param.requires_grad = False

            for m in self.named_modules():
                if 'model1' or 'model2' in m[0]:
                    continue
                if '' == m[0]:
                    continue
                
                m = m[1]
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    # print("module:", m)
                    # print("---------------------------------")
                    pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1_fc = self.fc1(deep_feats1)
        deep_feats2_fc = self.fc2(deep_feats2)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_expand = deep_feats1.unsqueeze(-1)
        deep_feats2_expand = deep_feats2.unsqueeze(-1)
        
        deep_feats1_expand = self.feat_dim(deep_feats1_expand)
        deep_feats2_expand = self.feat_dim(deep_feats2_expand)

        stacked_deep_feats = torch.stack((deep_feats1_expand, deep_feats2_expand), 2)

        weights = torch.matmul(stacked_deep_feats, self.weight_latent)
        weights = weights.squeeze(-1)
        weights = F.softmax(weights, -1)

        self.res_weights =  weights[:, :, 0]
        # print("res weights shape:", self.res_weights.shape)
        self.dense_weights = weights[:, :, 1] 
        # print("dense weights shape:", self.dense_weights.shape)
        res_weights_dict = {}
        dense_weights_dict = {}
        img_names = x[1]
        for ith_img in range(self.res_weights.shape[0]):
            res_weights_dict[img_names[ith_img]] = self.res_weights[ith_img]
            dense_weights_dict[img_names[ith_img]] = self.dense_weights[ith_img]

        self.res_weights_dict = add_key(self.res_weights_dict, res_weights_dict)
        self.dense_weights_dict = add_key(self.dense_weights_dict, dense_weights_dict)

        merged_feats = weights[:, :, 0] * deep_feats1 + weights[:, :, 1] * deep_feats2

        fc = self.fc(merged_feats)

        return fc, deep_feats1_fc, deep_feats2_fc

class ResNet17_4_Refined_W_Loss_New2(nn.Module):

    def __init__(self, num_mid_feats=32, num_classes=2, num_trans_dim=8):
            super(ResNet17_4_Refined_W_Loss_New2, self).__init__()


            res = models.resnet18(pretrained=True)
            res_list = list(res.children())
            res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=False)
            self.model1 = res
            self.fc1 = nn.Linear(num_mid_feats, num_classes)


            dense = models.densenet121(pretrained=True)
            dense.classifier = nn.Linear(1024, num_mid_feats, bias=True)
            self.model2 = dense
            self.fc2 = nn.Linear(num_mid_feats, num_classes)


            self.weight_latent = torch.nn.Parameter(torch.rand(size=(num_trans_dim,1)), requires_grad=True)

            self.fc = nn.Linear(num_mid_feats, num_classes)

            self.feat_dim = nn.Linear(1, num_trans_dim)

            # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

            # for param in self.model1.parameters():
            #     param.requires_grad = False

            # for param in self.model2.parameters():
            #     param.requires_grad = False

            for m in self.named_modules():
                if 'model1' or 'model2' in m[0]:
                    continue
                if '' == m[0]:
                    continue

                m = m[1]
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    # print("module:", m)
                    # print("---------------------------------")
                    pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1_fc = self.fc1(deep_feats1)
        deep_feats2_fc = self.fc2(deep_feats2)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_expand = deep_feats1.unsqueeze(-1)
        deep_feats2_expand = deep_feats2.unsqueeze(-1)

        deep_feats1_expand = self.feat_dim(deep_feats1_expand)
        deep_feats2_expand = self.feat_dim(deep_feats2_expand)

        stacked_deep_feats = torch.stack((deep_feats1_expand, deep_feats2_expand), 2)

        weights = torch.matmul(stacked_deep_feats, self.weight_latent)
        weights = weights.squeeze(-1)
        weights = F.softmax(weights, -1)


        merged_feats = weights[:, :, 0] * deep_feats1 + weights[:, :, 1] * deep_feats2

        fc = self.fc(merged_feats)

        return fc, deep_feats1_fc, deep_feats2_fc, weights



class ResNet17_4_Refined_W_Loss(nn.Module):

    def __init__(self, num_mid_feats=32, num_classes=2, num_trans_dim=8):
            super(ResNet17_4_Refined_W_Loss, self).__init__()

            res = models.resnet18(pretrained=True)
            res_list = list(res.children())
            res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=False)
            self.model1 = res
            self.fc1 = nn.Linear(num_mid_feats, num_classes)


            dense = models.densenet121(pretrained=True)
            dense.classifier = nn.Linear(1024, num_mid_feats, bias=True)
            self.model2 = dense
            self.fc2 = nn.Linear(num_mid_feats, num_classes)


            self.weight_latent = torch.nn.Parameter(torch.rand(size=(num_trans_dim,1)), requires_grad=True)

            self.fc = nn.Linear(num_mid_feats, num_classes)

            self.feat_dim = nn.Linear(1, num_trans_dim)

            # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

            # for param in self.model1.parameters():
            #     param.requires_grad = False

            # for param in self.model2.parameters():
            #     param.requires_grad = False

            for m in self.named_modules():
                if 'model1' or 'model2' in m[0]:
                    continue
                if '' == m[0]:
                    continue

                m = m[1]
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    # print("module:", m)
                    # print("---------------------------------")
                    pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1_fc = self.fc1(deep_feats1)
        deep_feats2_fc = self.fc2(deep_feats2)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_expand = deep_feats1.unsqueeze(-1)
        deep_feats2_expand = deep_feats2.unsqueeze(-1)

        deep_feats1_expand = self.feat_dim(deep_feats1_expand)
        deep_feats2_expand = self.feat_dim(deep_feats2_expand)

        stacked_deep_feats = torch.stack((deep_feats1_expand, deep_feats2_expand), 2)

        weights = torch.matmul(stacked_deep_feats, self.weight_latent)
        weights = weights.squeeze(-1)
        weights = F.softmax(weights, -1)

        merged_feats = weights[:, :, 0] * deep_feats1 + weights[:, :, 1] * deep_feats2

        fc = self.fc(merged_feats)

        return fc, deep_feats1_fc, deep_feats2_fc




class ResNet17_4_Refined_W_Loss_Single_Output(nn.Module):
    
    def __init__(self, num_mid_feats=32, num_classes=2, num_trans_dim=7):
            super(ResNet17_4_Refined_W_Loss_Single_Output, self).__init__()

            
            res = models.resnet18(pretrained=True)
            res_list = list(res.children())
            res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=False)
            self.model1 = res
            self.fc1 = nn.Linear(num_mid_feats, num_classes)


            dense = models.densenet121(pretrained=True)
            dense.classifier = nn.Linear(1024, num_mid_feats, bias=True)
            self.model2 = dense
            self.fc2 = nn.Linear(num_mid_feats, num_classes)


            self.weight_latent = torch.nn.Parameter(torch.rand(size=(num_trans_dim,1)), requires_grad=True)

            self.fc = nn.Linear(num_mid_feats, num_classes)

            self.feat_dim = nn.Linear(1, num_trans_dim)

            # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

            # for param in self.model1.parameters():
            #     param.requires_grad = False

            # for param in self.model2.parameters():
            #     param.requires_grad = False

            for m in self.named_modules():
                if 'model1' or 'model2' in m[0]:
                    continue
                if '' == m[0]:
                    continue
                
                m = m[1]
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    # print("module:", m)
                    # print("---------------------------------")
                    pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1_fc = self.fc1(deep_feats1)
        deep_feats2_fc = self.fc2(deep_feats2)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_expand = deep_feats1.unsqueeze(-1)
        deep_feats2_expand = deep_feats2.unsqueeze(-1)
        
        deep_feats1_expand = self.feat_dim(deep_feats1_expand)
        deep_feats2_expand = self.feat_dim(deep_feats2_expand)

        stacked_deep_feats = torch.stack((deep_feats1_expand, deep_feats2_expand), 2)

        weights = torch.matmul(stacked_deep_feats, self.weight_latent)
        weights = weights.squeeze(-1)
        weights = F.softmax(weights, -1)

        merged_feats = weights[:, :, 0] * deep_feats1 + weights[:, :, 1] * deep_feats2

        fc = self.fc(merged_feats)

        return fc



# 改为直接连接融合方式
class ResNet17_4_ConcatFusion_W_Loss(nn.Module):
    
    def __init__(self, num_mid_feats=32, num_classes=2, num_trans_dim=7):
            super(ResNet17_4_ConcatFusion_W_Loss, self).__init__()

            res = models.resnet18(pretrained=True)
            res_list = list(res.children())
            res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats, bias=False)
            self.model1 = res
            self.fc1 = nn.Linear(num_mid_feats, num_classes)


            dense = models.densenet121(pretrained=True)
            dense.classifier = nn.Linear(1024, num_mid_feats, bias=True)
            self.model2 = dense
            self.fc2 = nn.Linear(num_mid_feats, num_classes)


            self.fc = nn.Linear(num_mid_feats * 2, num_classes)

            for m in self.named_modules():
                if 'model1' or 'model2' in m[0]:
                    continue
                if '' == m[0]:
                    continue
                
                m = m[1]
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    # print("module:", m)
                    # print("---------------------------------")
                    pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)
        
        merged_feats = torch.cat((deep_feats1, deep_feats2), -1)
        fc = self.fc(merged_feats)

        deep_feats1_fc = self.fc1(deep_feats1)
        deep_feats2_fc = self.fc2(deep_feats2)

        return fc, deep_feats1_fc, deep_feats2_fc




class ResNet17_4_Refined(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2, num_trans_dim=3):
        super(ResNet17_4_Refined, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.weight_latent = torch.nn.Parameter(torch.rand(size=(num_trans_dim,1)), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        self.feat_dim = nn.Linear(1, num_trans_dim)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_expand = deep_feats1.unsqueeze(-1)
        deep_feats2_expand = deep_feats2.unsqueeze(-1)
        
        deep_feats1_expand = self.feat_dim(deep_feats1_expand)
        deep_feats2_expand = self.feat_dim(deep_feats2_expand)

        stacked_deep_feats = torch.stack((deep_feats1_expand, deep_feats2_expand), 2)

        weights = torch.matmul(stacked_deep_feats, self.weight_latent)
        weights = weights.squeeze(-1)
        weights = F.softmax(weights, -1)

        merged_feats = weights[:, :, 0] * deep_feats1 + weights[:, :, 1] * deep_feats2

        fc = self.fc(merged_feats)

        return fc



class ResNet17_4_Refined2(nn.Module):
    def __init__(self, model1, model2,  num_extra_feats=10, num_mid_feats=32, num_classes=2, num_trans_dim=4):
        super(ResNet17_4_Refined2, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.weight_latent = torch.nn.Parameter(torch.rand(size=(num_trans_dim,1)), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        self.feat_dim = nn.Linear(1, num_trans_dim)

        self.extra_fc = nn.Linear(num_extra_feats, num_mid_feats)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)    

        deep_feats3 = self.extra_fc(x[1])

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_expand = deep_feats1.unsqueeze(-1)
        deep_feats2_expand = deep_feats2.unsqueeze(-1)
        deep_feats3_expand = deep_feats3.unsqueeze(-1)
        
        deep_feats1_expand = self.feat_dim(deep_feats1_expand)
        deep_feats2_expand = self.feat_dim(deep_feats2_expand)
        deep_feats3_expand = self.feat_dim(deep_feats3_expand)

        stacked_deep_feats = torch.stack((deep_feats1_expand, deep_feats2_expand, deep_feats3_expand), 2)

        weights = torch.matmul(stacked_deep_feats, self.weight_latent)
        weights = weights.squeeze(-1)
        weights = F.softmax(weights, -1)

        merged_feats = weights[:, :, 0] * deep_feats1 + weights[:, :, 1] * deep_feats2 + weights[:, :, 2] * deep_feats3

        fc = self.fc(merged_feats)

        return fc



class ResNet17_4_Refined3(nn.Module):
    def __init__(self, model1, model2, model3, num_mid_feats=32, num_classes=2, num_trans_dim=4):
        super(ResNet17_4_Refined3, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model
        self.model3 = model3.model 

        self.weight_latent = torch.nn.Parameter(torch.rand(size=(num_trans_dim,1)), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        self.feat_dim = nn.Linear(1, num_trans_dim)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
            deep_feats3 = self.model3(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)    
            deep_feats3 = self.model3(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)
        deep_feats3 = deep_feats3.view(deep_feats3.size(0), -1)

        deep_feats1_expand = deep_feats1.unsqueeze(-1)
        deep_feats2_expand = deep_feats2.unsqueeze(-1)
        deep_feats3_expand = deep_feats3.unsqueeze(-1)
        
        deep_feats1_expand = self.feat_dim(deep_feats1_expand)
        deep_feats2_expand = self.feat_dim(deep_feats2_expand)
        deep_feats3_expand = self.feat_dim(deep_feats3_expand)

        stacked_deep_feats = torch.stack((deep_feats1_expand, deep_feats2_expand, deep_feats3_expand), 2)

        weights = torch.matmul(stacked_deep_feats, self.weight_latent)
        weights = weights.squeeze(-1)
        weights = F.softmax(weights, -1)

        merged_feats = weights[:, :, 0] * deep_feats1 + weights[:, :, 1] * deep_feats2 + weights[:, :, 2] * deep_feats3

        fc = self.fc(merged_feats)

        return fc



class ResNet17_4_Refined4(nn.Module):
    def __init__(self, model1, model2, model3, num_mid_feats=32, num_classes=2, num_trans_dim=4):
        super(ResNet17_4_Refined4, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model
        self.model3 = model3.model 

        self.weight_latent = torch.nn.Parameter(torch.rand(size=(num_trans_dim,1)), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        self.fc2 = nn.Linear(4, num_classes)

        self.feat_dim = nn.Linear(1, num_trans_dim)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)    
       
        deep_feats3 = self.model3(x[0]) #(B, 2)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_expand = deep_feats1.unsqueeze(-1)
        deep_feats2_expand = deep_feats2.unsqueeze(-1)
        
        deep_feats1_expand = self.feat_dim(deep_feats1_expand)
        deep_feats2_expand = self.feat_dim(deep_feats2_expand)

        stacked_deep_feats = torch.stack((deep_feats1_expand, deep_feats2_expand), 2)

        weights = torch.matmul(stacked_deep_feats, self.weight_latent)
        weights = weights.squeeze(-1)
        weights = F.softmax(weights, -1)

        merged_feats = weights[:, :, 0] * deep_feats1 + weights[:, :, 1] * deep_feats2 

        fc = self.fc(merged_feats)

        fc = torch.cat((fc, deep_feats3), -1)
        fc2 = self.fc2(fc)

        return fc2


class ResNet17_4_W_TRFM(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2):
        super(ResNet17_4_W_TRFM, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.weight_latent = torch.nn.Parameter(torch.rand(num_mid_feats), requires_grad=True)

        self.trfm = WindowAttention(dim=num_mid_feats, num_heads=1)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        stacked_feats = torch.stack((deep_feats1, deep_feats2), 0)
        trfmed_feats = self.trfm(stacked_feats)

        deep_feats1, deep_feats2 = trfmed_feats.unbind(0)

        weight_latent = self.weight_latent.view(-1, 1) # (32,1)
        feats = torch.stack((deep_feats1, deep_feats2), 1) # (5,2,32)
        
        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1 * weight_0 + deep_feats2 * weight_1

        fc = self.fc(merged_feats)

        return fc





class ResNet17_44(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2):
        super(ResNet17_44, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model


        self.weight_latent = torch.nn.Parameter(torch.rand(num_mid_feats), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        weight_latent = self.weight_latent.view(-1, 1) # (32,1)
        feats = torch.stack((deep_feats1, deep_feats2), 1) # (5,2,32)
        
        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1 * weight_0 + deep_feats2 * weight_1

        fc = self.fc(merged_feats)

        return fc




class ResNet17_4_Plus(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2):
        super(ResNet17_4_Plus, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.weight_latent = torch.nn.Parameter(torch.rand(num_mid_feats), requires_grad=True)

        self.weight_latent2 = torch.nn.Parameter(torch.rand(num_mid_feats), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats * 2, num_classes)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        weight_latent = self.weight_latent.view(-1, 1) # (32,1)
        feats = torch.stack((deep_feats1, deep_feats2), 1) # (5,2,32)
        
        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats_0 = deep_feats1 * weight_0 + deep_feats2 * weight_1


        weight_latent1 = self.weight_latent.view(-1, 1) # (32,1)
        feats1 = torch.stack((deep_feats1, deep_feats2), 1) # (5,2,32)
        
        weight1 = torch.matmul(feats1, weight_latent1) # (5, 2, 1)
        weight1 = weight.view(weight1.shape[0], weight1.shape[1]) # (5, 2)

        weight1 = F.softmax(weight1, -1) 

        weight1_0 = weight1[:, 0].view(-1, 1)
        weight1_1 = weight1[:, 1].view(-1, 1)

        merged_feats_1 = deep_feats1 * weight1_0 + deep_feats2 * weight1_1

        merged_feats = torch.cat((merged_feats_0, merged_feats_1), -1)

        fc = self.fc(merged_feats)

        return fc



class ResNet17_4_Plus2(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2, num_random_feats=5):
        super(ResNet17_4_Plus2, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.rand_latent_1 = torch.nn.Parameter(torch.rand(num_random_feats), requires_grad=True)
        self.rand_latent_2 = torch.nn.Parameter(torch.rand(num_random_feats), requires_grad=True)

        self.weight_latent = torch.nn.Parameter(torch.rand((num_mid_feats + num_random_feats)), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats + num_random_feats, num_classes)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1 = torch.cat((deep_feats1, self.rand_latent_1.repeat(deep_feats1.shape[0], 1)), -1)
        deep_feats2 = torch.cat((deep_feats2, self.rand_latent_2.repeat(deep_feats2.shape[0], 1)), -1)

        weight_latent = self.weight_latent.view(-1, 1) # (37,1)
        feats = torch.stack((deep_feats1, deep_feats2), 1) # (5,2,37)
        
        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1 * weight_0 + deep_feats2 * weight_1

        fc = self.fc(merged_feats)

        return fc



class ResNet17_4_Plus3(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2, num_random_feats=1):
        super(ResNet17_4_Plus3, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.rand_latent = torch.nn.Parameter(torch.rand(num_random_feats), requires_grad=True)

        self.weight_latent = torch.nn.Parameter(torch.rand(num_mid_feats), requires_grad=True)

        self.fc = nn.Linear(num_mid_feats + num_random_feats, num_classes)

        # self.random_v = torch.nn.Parameter(torch.randn(5), requires_grad=True)

        # for param in self.model1.parameters():
        #     param.requires_grad = False

        # for param in self.model2.parameters():
        #     param.requires_grad = False

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        weight_latent = self.weight_latent.view(-1, 1) # (32,1)
        feats = torch.stack((deep_feats1, deep_feats2), 1) # (5,2,32)
        
        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1 * weight_0 + deep_feats2 * weight_1

        merged_feats = torch.cat((merged_feats, self.rand_latent.repeat(merged_feats.shape[0], 1)), -1)

        fc = self.fc(merged_feats)

        return fc



class ResNet17_5(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2, num_weight_dim = 10):
        super(ResNet17_5, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.weight_latent = torch.nn.Parameter(torch.rand(num_weight_dim), requires_grad=True)

        self.fc1 = nn.Linear(num_mid_feats, num_weight_dim)
        self.fc2 = nn.Linear(num_mid_feats, num_weight_dim)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_w = self.fc1(deep_feats1)
        deep_feats2_w = self.fc2(deep_feats2)

        weight_latent = self.weight_latent.view(-1, 1) # (10,1)
        feats = torch.stack((deep_feats1_w, deep_feats2_w), 1) # (5,2,10)

        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1 * weight_0 + deep_feats2 * weight_1

        fc = self.fc(merged_feats)

        return fc





class ResNet17_5(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2, num_weight_dim = 10):
        super(ResNet17_5, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.weight_latent = torch.nn.Parameter(torch.rand(num_weight_dim), requires_grad=True)

        self.fc1 = nn.Linear(num_mid_feats, num_weight_dim)
        self.fc2 = nn.Linear(num_mid_feats, num_weight_dim)

        self.fc = nn.Linear(num_mid_feats, num_classes)

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_w = self.fc1(deep_feats1)
        deep_feats2_w = self.fc2(deep_feats2)

        weight_latent = self.weight_latent.view(-1, 1) # (10,1)
        feats = torch.stack((deep_feats1_w, deep_feats2_w), 1) # (5,2,10)

        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1 * weight_0 + deep_feats2 * weight_1

        fc = self.fc(merged_feats)

        return fc



class ResNet17_6(nn.Module):
    def __init__(self, model1, model2,  num_mid_feats=32, num_classes=2, num_weight_dim = 10):
        super(ResNet17_6, self).__init__()

        self.model1 = model1.model
        self.model2 = model2.model

        self.weight_latent = torch.nn.Parameter(torch.rand(num_weight_dim), requires_grad=True)

        self.fc1 = nn.Linear(num_mid_feats, num_weight_dim)
        self.fc2 = nn.Linear(num_mid_feats, num_weight_dim)

        self.fc = nn.Linear(num_weight_dim, num_classes)

        for m in self.named_modules():
            if 'model1' or 'model2' in m[0]:
                continue
            if '' == m[0]:
                continue
            
            m = m[1]
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats1 = self.model1(x[0])
            deep_feats2 = self.model2(x[0])
        else:
            deep_feats1 = self.model1(x)
            deep_feats2 = self.model2(x)

        deep_feats1 = deep_feats1.view(deep_feats1.size(0), -1)
        deep_feats2 = deep_feats2.view(deep_feats2.size(0), -1)

        deep_feats1_w = self.fc1(deep_feats1)
        deep_feats2_w = self.fc2(deep_feats2)

        weight_latent = self.weight_latent.view(-1, 1) # (10,1)
        feats = torch.stack((deep_feats1_w, deep_feats2_w), 1) # (5,2,10)

        weight = torch.matmul(feats, weight_latent) # (5, 2, 1)
        weight = weight.view(weight.shape[0], weight.shape[1]) # (5, 2)

        weight = F.softmax(weight, -1) 

        weight_0 = weight[:, 0].view(-1, 1)
        weight_1 = weight[:, 1].view(-1, 1)

        merged_feats = deep_feats1_w * weight_0 + deep_feats2_w * weight_1

        fc = self.fc(merged_feats)

        return fc



class ResNet18(nn.Module):
    def __init__(self,  num_mid_feats=32, num_classes=2):
        super(ResNet18, self).__init__()

        res = models.resnet50(pretrained=True)
        res_list = list(res.children())
        res.fc = nn.Linear(res_list[-1].in_features, num_mid_feats)
        self.model = res
       
        self.fc = nn.Linear(num_mid_feats , num_classes)

        for m in self.named_modules():
            if 'model' in m[0] and 'model.fc' not in m[0]:
                continue
            if '' == m[0]:
                continue
            

            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            deep_feats = self.model(x[0])
        else:
            deep_feats = self.model(x)

        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        fc = self.fc(deep_feats)

        return fc



class DenseNet2(nn.Module):
    def __init__(self, num_mid_features = 32, num_classes=2):
        super(DenseNet2, self).__init__()
        dense = models.densenet121(pretrained=True)
        dense.classifier = nn.Linear(1024, num_mid_features, bias=True)

        self.model = dense
        self.fc = nn.Linear(num_mid_features, num_classes)

        for m in self.named_modules():

            if 'model' in m[0] and 'model.classifier' not in m[0]:
                continue
            if '' == m[0]:
                continue
            
            # if m[0] == 'model.classifier':
            #     print('yes')
            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)

        res = res.view(res.shape[0], -1)
        res = self.fc(res)
        return res



class DenseNet2_W_WeightPool(nn.Module):
    def __init__(self, num_mid_features = 32, num_classes=2, num_trans_dim=50):
        super(DenseNet2_W_WeightPool, self).__init__()
        dense = models.densenet121(pretrained=True)
        dense_list = list(dense.features.children())
        new_dense_list = dense_list
        new_dense_list.append(nn.AdaptiveAvgPool2d(output_size=(2,2)))
        # new_dense_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # new_dense_list.append(WeightedPool4(num_trans_dim=num_trans_dim))
        new_dense_list.append(WeightedPool2(num_trans_dim=num_trans_dim))
        new_dense_list.append(nn.Linear(1024, num_classes, bias=True))
        # new_dense_list.append(nn.Linear(1024, num_mid_features, bias=True))

        self.model = nn.Sequential(*new_dense_list)
        # self.fc = nn.Linear(num_mid_features, num_classes)

        for m in self.named_modules():

            if 'model' in m[0] and 'model.classifier' not in m[0]:
                continue
            if '' == m[0]:
                continue
            
            # if m[0] == 'model.classifier':
            #     print('yes')
            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)

        res = res.view(res.shape[0], -1)
        # res = self.fc(res)
        return res



class DenseNet2_W_WeightPool2(nn.Module):
    def __init__(self, num_mid_features = 32, num_classes=2, num_trans_dim=50):
        super(DenseNet2_W_WeightPool2, self).__init__()
        dense = models.densenet121(pretrained=True)
        dense_list = list(dense.features.children())
        new_dense_list = dense_list
        new_dense_list.append(WeightedPool2(num_trans_dim=num_trans_dim))
        new_dense_list.append(nn.Linear(1024, num_classes, bias=True))
        # new_dense_list.append(nn.Linear(1024, num_mid_features, bias=True))

        self.model = nn.Sequential(*new_dense_list)
        # self.fc = nn.Linear(num_mid_features, num_classes)

        for m in self.named_modules():

            if 'model' in m[0] and 'model.classifier' not in m[0]:
                continue
            if '' == m[0]:
                continue
            
            # if m[0] == 'model.classifier':
            #     print('yes')
            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)

        res = res.view(res.shape[0], -1)
        # res = self.fc(res)
        return res



class DenseNet2_W_WeightPool_3(nn.Module):
    def __init__(self, num_mid_features = 32, num_classes=2, num_trans_dim=50):
        super(DenseNet2_W_WeightPool_3, self).__init__()
        dense = models.densenet121(pretrained=True)
        dense_list = list(dense.features.children())
        new_dense_list = dense_list
        new_dense_list.append(nn.AdaptiveAvgPool2d(output_size=(2,2)))
        # new_dense_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # new_dense_list.append(WeightedPool4(num_trans_dim=num_trans_dim))
        new_dense_list.append(WeightedPool2(num_trans_dim=num_trans_dim))
        new_dense_list.append(nn.Linear(1024, num_mid_features, bias=True))

        self.model = nn.Sequential(*new_dense_list)
        self.fc = nn.Linear(num_mid_features, num_classes)

        for m in self.named_modules():

            if 'model' in m[0] and 'model.classifier' not in m[0]:
                continue
            if '' == m[0]:
                continue
            
            # if m[0] == 'model.classifier':
            #     print('yes')
            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)

        res = res.view(res.shape[0], -1)
        res = self.fc(res)
        return res




class DenseNet3(nn.Module):
    def __init__(self, num_mid_features = 32, num_classes=2):
        super(DenseNet3, self).__init__()
        dense = models.densenet169(pretrained=True)
        dense.classifier = nn.Linear(1664, num_mid_features, bias=True)

        self.model = dense
        self.fc = nn.Linear(num_mid_features, num_classes)

        for m in self.named_modules():

            if 'model' in m[0] and 'model.classifier' not in m[0]:
                continue
            if '' == m[0]:
                continue
            
            # if m[0] == 'model.classifier':
            #     print('yes')
            m1 = m[1]
            if isinstance(m1, nn.Conv2d):
                nn.init.kaiming_normal_(m1.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)
            elif isinstance(m1, nn.Linear):
                # print(m[0])
                nn.init.xavier_normal_(m1.weight)
                nn.init.constant_(m1.bias, 0)
            else:
                # print("module:", m)
                # print("---------------------------------")
                pass


    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)

        res = res.view(res.shape[0], -1)
        # res = self.fc(res)
        return res




class ASPP(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 32, num_classes = 2):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.fc = nn.Linear(out_channels, num_classes)

      
    def forward(self, x):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        if type(x) == tuple:
            feature_map = x[0]
        else:
            feature_map = x

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_1x1 = self.avg_pool(out_1x1)

        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = self.avg_pool(out_3x3_1)

        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = self.avg_pool(out_3x3_2)

        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = self.avg_pool(out_3x3_3)

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))

        out = (out_1x1 + out_3x3_1 + out_3x3_2 + out_3x3_3 + out_img) / 5

        out = out.view(out.shape[0], -1)

        out = self.fc(out)

        return out




class ASPP2(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 32, num_classes = 2):
        super(ASPP2, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.fc = nn.Linear(out_channels * 5, num_classes)

      
    def forward(self, x):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        if type(x) == tuple:
            feature_map = x[0]
        else:
            feature_map = x

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_1x1 = self.avg_pool(out_1x1)

        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = self.avg_pool(out_3x3_1)

        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = self.avg_pool(out_3x3_2)

        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = self.avg_pool(out_3x3_3)

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))

        out = torch.cat((out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img), -1)

        out = out.view(out.shape[0], -1)

        out = self.fc(out)

        return out


class InceptionNet(nn.Module):

    def __init__(self, num_classes = 2):
        super(InceptionNet, self).__init__()
        inception  = models.inception_v3(num_classes=num_classes)
        self.model = inception


    def forward(self, x):
        if type(x) == tuple: 
            return self.model(x[0])
        else:
            return self.model(x)



class InceptionNet2(nn.Module):

    def __init__(self, num_classes = 2, num_mid_features=32):
        super(InceptionNet2, self).__init__()
        inception  = models.inception_v3(num_classes=num_mid_features)
        self.model = inception
        self.fc1 = nn.Linear(num_mid_features, num_classes, bias=True)
        self.fc2 = nn.Linear(num_mid_features, num_classes, bias=True)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        res1, res2 = self.model(x[0])
        res1 = self.fc1(res1)
        res2 = self.fc2(res2)
        return res1, res2



class VggNet(nn.Module):

    def __init__(self, num_classes = 2):
        super(VggNet, self).__init__()
        vgg  = models.vgg11(pretrained=True)
        vgg.classifier = nn.Linear(25088, num_classes, bias=True)
        self.model = vgg

        # for m in self.children():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)
        
        # nn.init.xavier_normal_(self.model.classifier.weight)
        # nn.init.constant_(self.model.classifier.bias, 0)

        for m in self.named_modules():
    

            m1 = m[1]
          
            if isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                # print("yes")
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)

        nn.init.normal_(self.model.classifier.weight, 0, 0.01)
        nn.init.constant_(self.model.classifier.bias, 0)

    


    def forward(self, x):
        # print('shape:', x[0].shape)
        if type(x) == tuple: 
            res = self.model(x[0])
        else:
            res = self.model(x)
        return res


class VggNet16(nn.Module):

    def __init__(self, num_classes = 2):
        super(VggNet16, self).__init__()
        vgg  = models.vgg16(pretrained=True)
        vgg.classifier[-1] = nn.Linear(4096, num_classes, bias=True)
        self.model = vgg

        for m in self.named_modules():


            m1 = m[1]

            if isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                # print("yes")
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)

        nn.init.normal_(self.model.classifier[-1].weight, 0, 0.01)
        nn.init.constant_(self.model.classifier[-1].bias, 0)


    def forward(self, x):
        # print('shape:', x[0].shape)
        if type(x) == tuple: 
            res = self.model(x[0])
        else:
            res = self.model(x)        
        return res


class VggNet2(nn.Module):

    def __init__(self, num_classes = 2, num_mid_features=32):
        super(VggNet2, self).__init__()
        vgg  = models.vgg11(pretrained=True)
        vgg.classifier = nn.Linear(25088, num_mid_features, bias=True)
        self.model = vgg
        self.fc = nn.Linear(num_mid_features, num_classes, bias=True)

        for m in self.named_modules():

            m1 = m[1]

            if isinstance(m1, (nn.BatchNorm2d, nn.GroupNorm)):
                # print("yes")
                nn.init.constant_(m1.weight, 1)
                nn.init.constant_(m1.bias, 0)

        nn.init.normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        

    def forward(self, x):
        # print('shape:', x[0].shape)
        res= self.model(x[0])
        res = self.fc(res)
        return res



class VggNet16_2(nn.Module):

    def __init__(self, num_classes = 2, num_mid_features=32):
        super(VggNet16_2, self).__init__()
        vgg  = models.vgg11(pretrained=True)
        vgg.classifier[-1] = nn.Linear(4096, num_mid_features, bias=True)
        self.model = vgg
        self.fc = nn.Linear(num_mid_features, num_classes, bias=True)

        # for m in self.children():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias)
        nn.init.xavier_normal_(vgg.classifier[-1].weight)
        nn.init.constant_(vgg.classifier[-1].bias, 0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        

    def forward(self, x):
        # print('shape:', x[0].shape)
        res= self.model(x[0])
        res = self.fc(res)
        return res


class EfficientNet(nn.Module):

    def __init__(self, num_classes = 2):
        super(EfficientNet, self).__init__()
        efficient_net = models.efficientnet_b0(pretrained=True)
        efficient_net.classifier[1] = nn.Linear(1280, num_classes)
        self.model = efficient_net
        
        nn.init.xavier_normal_( efficient_net.classifier[1].weight)
        nn.init.constant_( efficient_net.classifier[1].bias, 0)


    def forward(self, x):
        if type(x) == tuple:
            return self.model(x[0])
        else:
            return self.model(x)



class EfficientNet2(nn.Module):

    def __init__(self, num_classes = 2, num_mid_features = 32):
        super(EfficientNet2, self).__init__()
        efficient_net = models.efficientnet_b0(pretrained=True)
        efficient_net.classifier[1] = nn.Linear(1280, num_mid_features)
        self.model = efficient_net
        self.fc = nn.Linear(num_mid_features, num_classes, bias=True)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_normal_( efficient_net.classifier[1].weight)
        nn.init.constant_( efficient_net.classifier[1].bias, 0)


    def forward(self, x):
        if type(x) == tuple:
            res = self.model(x[0])
        else:
            res = self.model(x)
        res = self.fc(res)
        return res


class AlexNet(nn.Module):

    def __init__(self, num_classes = 2):
        super(AlexNet, self).__init__()
        net = models.alexnet(pretrained=True)
        net.classifier[-1] = nn.Linear(4096, num_classes)
        self.model = net
        
        nn.init.xavier_normal_( net.classifier[-1].weight)
        nn.init.constant_( net.classifier[-1].bias, 0)
    

    def forward(self, x):
        if type(x) == tuple:
            return self.model(x[0])
        else:
            return self.model(x)


class AlexNet2(nn.Module):

    def __init__(self, num_classes = 2, num_mid_features = 32):
        super(AlexNet2, self).__init__()
        net = models.alexnet(pretrained=True)
        net.classifier[-1] = nn.Linear(4096, num_mid_features)
        self.model = net
        self.fc = nn.Linear(num_mid_features, num_classes)
        
        nn.init.xavier_normal_( net.classifier[-1].weight)
        nn.init.constant_( net.classifier[-1].bias, 0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)


    def forward(self, x):
        res = self.model(x[0])
        res = self.fc(res)
        return res




class FDFFE(nn.Module):
    
    def __init__(self, num_mid_features = 32, num_classes = 2):
        super().__init__()
        self.fc1 = nn.Linear(10, num_mid_features, bias=True)
        self.fc2 = nn.Linear(num_mid_features, num_classes, bias=True)

        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        if type(x) == tuple:
            res = self.fc1(x[1])
        else:
            res = self.fc1(x)
        res = self.fc2(res)

        return res



if __name__ == '__main__':
    pass

    # x = torch.randn(5, 3, 128, 128)
    # model = VggNet()
    # print(model((x,)).shape)

    # x = torch.randn(5, 3, 512, 512)
    # model = InceptionNet()
    # print(model((x,)))

    # weighted_pool = WeightedPool2(num_trans_dim=5)
    # x = torch.rand(5, 1024, 12, 12)
    # print(weighted_pool(x).shape)



    # m = ResNet5()

    # m = ResNet10()

    # x = torch.randn(5,3,256,256)


    # m = ResNet13()
    # print(m(x).shape)
    # m2 = ResNet14(m)
    # x1 = torch.randn(5,10)
    # print(m2((x,x1)).shape)
  
    # x = torch.randn(5, 3, 128, 128)
    # model = DenseNet2(num_classes=2,num_mid_features=32)
    # print(model(x).shape)

    # x = torch.randn(5, 3, 128, 128)
    
    # model = ASPP(in_channels=3, out_channels=32)
    # print(model(x).shape)

    # model2 = ASPP2(in_channels=3, out_channels=32)
    # print(model2(x).shape)



    # model = NetMerge()
    # x = torch.randn(5, 3, 256, 256)
    # print(model(x).shape)


    # backbone = NetMerge(has_second_fc=False)
    # model = NetMergeWExtra(backbone=backbone, num_extra_feats=25, num_mid_feats=1000, num_classes=2)
    # x = torch.randn(5, 3, 256, 256)
    # y = torch.randn(5, 25)
    # outputs = model.forward((x,y))
    # print(outputs.shape)
 

    # backbone = NetMerge(has_second_fc=False)
    # model = NetMergeWExtraWLatent(backbone=backbone, num_extra_feats=25, 
    #                     latent_from_dim=10, latent_to_dim=100,
    #                     num_mid_feats=1000, num_classes=2)
    # x = torch.randn(5, 3, 256, 256)
    # y = torch.randn(5, 25)
    # outputs = model.forward((x,y))
    # print(outputs.shape)


    # x = torch.randn(5, 3, 128, 128)
    # model = Net32Plus()
    # print(model(x).shape)
    
    
    # model = models.densenet121(pretrained=True)
    # model = Net(model)
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net2(model)
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net3(model)
    # # x = torch.randn(10, 3, 250, 250)
    # x =  torch.randn(10, 3, 289,289)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net4(model)
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net5(model)
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # z = torch.randn(10, 1024)
    # outputs = model.forward((x, y, z))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net6(model)
    # print("model6")
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # z = torch.randn(10, 1024)
    # outputs = model.forward((x, y, z))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net7(model)
    # print("model7")
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # z = torch.randn(10, 1024)
    # outputs = model.forward((x, y, z))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net8(model)
    # print("model8")
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # z = torch.randn(10, 1024)
    # outputs = model.forward((x, y, z))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net9(model)
    # print("model9")
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net10(model)
    # print("model10")
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 25)
    # z = torch.randn(10, 1024)
    # outputs = model.forward((x, y, z))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net11(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 250, 250)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net12(model)
    # print("model12")
    # x = torch.randn((10, 3, 256, 256), requires_grad=True, dtype=torch.double)
    # y = torch.randn((10, 25), requires_grad=True, dtype=torch.double)
    # z = torch.randn((10, 3, 256, 256), requires_grad=True, dtype=torch.double)
    # outputs = model.forward((x, y, z))
    # print(outputs.shape)
    # print(outputs)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # pass

    # model = models.densenet121(pretrained=True)
    # model = Net4(model)
    # x = torch.randn(10, 3, 124, 124)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net13(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 124, 124)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net14(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 124, 124)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model = Net15(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model = Net17(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # dense_model = models.densenet121(pretrained=True)
    # dense_model = Net32(dense_model)
    # model = Net17Simple(model, dense_model, num_extra_feats=10)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)



    # model = models.resnet50(pretrained=True)
    # res_model = Net32(model)
    # dense_model = models.densenet121(pretrained=True)
    # dense_model = Net32(dense_model)
    # model = Net17Simple2(res_model, dense_model, num_classes=10)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)


    # model = models.resnet50(pretrained=True)
    # res_model = Net32(model)
    # dense_model = models.densenet121(pretrained=True)
    # dense_model = Net32(dense_model)
    # model = Net17Simple3(res_model, dense_model, num_classes=2)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)


    # model = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32],
    #                             window_size=7, drop_path_rate=0.5, num_classes=1024)
    # model = Net16(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model = Net18(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model1 = models.resnet50(pretrained=True)
    # model2 = models.resnet50(pretrained=True)
    # model = Net19(model1, model2, num_extra_feats=10)
    # x1 = torch.randn(10, 3, 256, 256)
    # x2 = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x1, y, x2))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model = Net20(model,num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((y, x))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model = Net20(model,num_extra_feats=10)
    # x = torch.randn(10, 3, 16, 16)
    # y = torch.randn(10, 10)
    # outputs = model.forward((y, x))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d(4,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net17(model, num_extra_feats=10)
    # x = torch.randn(10, 4, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d( 3 * 64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net23(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)


    # model = models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d( 3 * 16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net25(model, num_extra_feats=10)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d( 3 * 16, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net27(model, num_extra_feats=10)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net28(model, num_extra_feats=10)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)
    # pass

 
    # renset50
    # model = ACmix_ResNet(layers=[3,4,6,3], num_classes=1000) 
    # resnet38
    # model = ACmix_ResNet(layers=[3,3,3,3], num_classes=1000)
    # resnet18
    # model = ACmix_ResNet2(layers=[2,2,2,2], num_classes=1000)
    # model = Net29(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)


    # model = models.resnet50(pretrained=True)
    # model = Net30(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # model = Net31(model, num_extra_feats=10)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.densenet121(pretrained=True)
    # model = Net32(model)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 10)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # dense_model = models.densenet121(pretrained=True)
    # dense_model = Net32(dense_model)

    # model = Net33(model, dense_model, num_extra_feats=25)
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # dense_model = models.densenet121(pretrained=True)
    # dense_model = Net32(dense_model)
    # model = Net34(model, dense_model, num_extra_feats=25)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # dense_model = models.densenet121(pretrained=True)
    # dense_model = Net32(dense_model)
    # model = Net35(model, dense_model, num_extra_feats=25)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)

    # model = models.resnet50(pretrained=True)
    # dense_model = models.densenet121(pretrained=True)
    # dense_model = Net32(dense_model)
    # model = Net35_3(model, dense_model, num_extra_feats=25)

    # for parm in model.dense_layer.parameters():
    #     if not parm.requires_grad:
    #         print('parm:', parm)

    # for m in model.children():
    #     if not m.requires_grad_:
    #         print('m', m)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)


    # model = models.resnet50(pretrained=True)
    # model = Net36(model, num_extra_feats=25)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    
    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x, y))
    # print(outputs.shape)
    # print(outputs)
    
    # model = models.resnet50(pretrained=True)
    # model = NetWrapper(model, 2)

    # x = torch.randn(10, 3, 256, 256)
    # y = torch.randn(10, 25)
    # outputs = model.forward((x,y))
    # print(outputs.shape)
    # print(outputs)

    pass


