# -*- encoding: utf-8 -*-
"""
@File    : resnet50_plus.py
@Time    : 12/22/2021 2:42 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import torchvision.models as models
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=1000, num_classes=2):
        super(Net, self).__init__()
        # -2表示去掉model的后两层
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(2048 + num_extra_feats, num_mid_feats)
        self.fc2 = nn.Linear(num_mid_feats, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)
        merged_feats = torch.cat([deep_feats, x[1]], axis=1)

        # 将一个多行的Tensor,拼接成一行,-1指在不告诉函数有多少列
        merged_feats = merged_feats.view(merged_feats.size(0), -1)
        fc1 = self.fc1(merged_feats)
        fc2 = self.fc2(fc1)
        return fc2


class Net2(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_mid_feats=2048, num_classes=2):
        super(Net2, self).__init__()
        # -2表示去掉model的后两层
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
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


class Net3(nn.Module):
    def __init__(self, model, num_classes=2):
        super(Net3, self).__init__()
        # -2表示去掉model的后两层
        self.res_layer = model
        num_feats = list(model.children())[-1].out_features
        self.append_fc = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        # print(deep_feats.shape)

        result = self.append_fc(deep_feats)
        return result


class Net4(nn.Module):
    def __init__(self, model, num_extra_feats=25, num_classes=2):
        super(Net4, self).__init__()
        # -2表示去掉model的后两层
        self.res_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(num_extra_feats, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        deep_feats = self.res_layer(x[0])
        deep_feats = deep_feats.view(deep_feats.size(0), -1)

        extra_feats = x[1]
        extra_feats = self.fc1(extra_feats)
        merged_feats = torch.add(deep_feats, extra_feats)

        fc2 = self.fc2(merged_feats)
        return fc2


vgg = models.resnet50(pretrained=True)
model = Net(vgg)
x = torch.randn(10, 3, 250, 250)
y = torch.randn(10, 25)
outputs = model.forward((x, y))
print(outputs.shape)
print(outputs)
#
# model2 = Net2(vgg)
# x = torch.randn(10, 3, 250, 250)
# y = torch.randn(10, 25)
# outputs = model2.forward((x, y))
# print(outputs.shape)
# print(outputs)

# model = Net3(vgg)
# x = torch.randn(10, 3, 250, 250)
# y = torch.randn(10, 25)
# outputs = model.forward((x, y))
# print(outputs.shape)
# print(outputs)


# model = Net4(vgg)
# x = torch.randn(10, 3, 250, 250)
# y = torch.randn(10, 25)
# outputs = model.forward((x, y))
# print(outputs.shape)
# print(outputs)

