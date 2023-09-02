# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 11/30/2021 4:45 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
from collections import OrderedDict

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from dataset import CustomData
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import time

# def train_model(model, dataloader, criterion, optimizer, scheduler,
#                 num_epochs, use_gpu, img_datasets):
#
#     best_acc = 0.0
#     best_model_wts = model.state_dict()
#     best_epoch = -1
#     since = time.time()
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs))
#         print('-' * 20)
#
#         if epoch % 5 == 0:
#             running_stages = ['train', 'val']
#         else:
#             running_stages = ['train']
#
#         running_loss = 0.0
#         running_corrects = 0.0
#         for phase in running_stages:
#
#             if phase == 'train':
#                 model.train(True)
#             else:
#                 model.train(False)
#
#             for i, (inputs, labels) in enumerate(dataloader[phase]):
#                 print("i is:", i)
#
#                 if use_gpu:
#                     inputs = inputs.cuda()
#                     labels = labels.cuda()
#
#                 optimizer.zero_grad()
#
#                 outputs = model.forward(inputs)
#                 # print(outputs)
#                 # print(type(outputs))
#                 # print(outputs.shape)
#                 # print(dir(outputs))
#                 # print(outputs.data)
#                 _, preds = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#                     scheduler.step()
#
#                 # print(loss)
#                 # print(type(loss))
#                 # print(loss.shape)
#                 # print(loss.data == loss)
#                 running_loss += loss.data
#                 running_corrects += torch.sum(preds == labels.data).to(torch.float32)
#
#             epoch_loss = running_loss / len(image_datasets[phase])
#             epoch_acc = running_corrects / len(image_datasets[phase])
#
#             print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))
#
#             if phase == 'train':
#                 torch.save(model, './pkls/epoch_' + str(epoch) + '.pkl')
#
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_model_wts = model.state_dict()
#                 best_acc = epoch_acc
#
#                 best_epoch = epoch
#
#     elasped_time = time.time() - since
#     print('Training and Validating finished in: {:.0f} m {:.0f}s'.format(
#         elasped_time // 60, elasped_time % 60
#     ))
#     print('Best Acc:{:.4f}'.format(best_acc))
#     model = model.load_state_dict(best_model_wts)
#     torch.save(model, './pkls/best_model.pkl')
#     return model

# def train_model(model, dataloader, criterion, optimizer, scheduler,
#                 num_epochs, use_gpu, img_datasets):
#
#     best_acc = 0.0
#     best_model_wts = model.state_dict()
#     best_epoch = -1
#     since = time.time()
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs))
#         print('-' * 20)
#
#         if epoch % 5 == 0:
#             running_stages = ['train', 'val']
#         else:
#             running_stages = ['train']
#
#         for phase in running_stages:
#
#             if phase == 'train':
#                 scheduler.step()
#                 model.train(True)
#             else:
#                 model.train(False)
#
#             running_loss = 0.0
#             running_corrects = 0.0
#
#             for i, (inputs, labels) in enumerate(dataloader[phase]):
#                 if use_gpu:
#                     inputs = inputs.cuda()
#                     labels = labels.cuda()
#
#                 optimizer.zero_grad()
#
#                 outputs = model.forward(inputs)
#                 # print(outputs)
#                 # print(type(outputs))
#                 # print(outputs.shape)
#                 # print(dir(outputs))
#                 # print(outputs.data)
#                 _, preds = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#
#                 # print(loss)
#                 # print(type(loss))
#                 # print(loss.shape)
#                 # print(loss.data == loss)
#                 running_loss += loss.data
#                 running_corrects += torch.sum(preds == labels.data).to(torch.float32)
#
#             epoch_loss = running_loss / len(image_datasets[phase])
#             epoch_acc = running_corrects / len(image_datasets[phase])
#
#             print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))
#
#             if phase == 'train':
#                 torch.save(model, './pkls/epoch_' + str(epoch) + '.pkl')
#
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_model_wts = model.state_dict()
#                 best_acc = epoch_acc
#                 best_epoch = epoch
#
#     elasped_time = time.time() - since
#     print('Training and Validating finished in: {:.0f} m {:.0f}s'.format(
#         elasped_time // 60, elasped_time % 60
#     ))
#     print('Best Acc:{:.4f}'.format(best_acc))
#     model = model.load_state_dict(best_model_wts)
#     torch.save(model, './pkls/best_model.pkl')
#     return model
from train2 import get_consistent_mean_and_std


def train_model(model, dataloader, criterion, optimizer, scheduler,
                num_epochs, use_gpu, img_datasets, model_name=''):

    best_acc = 0.0
    best_model_wts = model.state_dict()
    best_epoch = -1
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 20)

        if epoch % 5 == 0:
            running_stages = ['train', 'val']
        else:
            running_stages = ['train']

        for phase in running_stages:

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for i, (inputs, labels) in enumerate(dataloader[phase]):
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                # print(outputs)
                # print(type(outputs))
                # print(outputs.shape)
                # print(dir(outputs))
                # print(outputs.data)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # print(loss)
                # print(type(loss))
                # print(loss.shape)
                # print(loss.data == loss)
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects / len(image_datasets[phase])

            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                torch.save(model, './deep_model2/pkls/' + str(model_name) + '_epoch_' + str(epoch) + '.pkl')

            if phase == 'val' and epoch_acc > best_acc:
                best_model_wts = model.state_dict()
                best_acc = epoch_acc
                best_epoch = epoch

    elasped_time = time.time() - since
    print('Training and Validating finished in: {:.0f} m {:.0f}s'.format(
        elasped_time // 60, elasped_time % 60
    ))
    print('Best Acc:{:.4f}'.format(best_acc))
    model = model.load_state_dict(best_model_wts)
    torch.save(model, './deep_model2/pkls/best_' + str(model_name) + '.pkl')
    return model


def get_normalized_mean_and_std():
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
    image_datasets = {x: CustomData(img_path='./split_data/' + x,
                                    data_transforms=data_transforms,
                                    dataset_split_name=x) for x in ['train', 'val']}
    train_data_loaders = DataLoader(image_datasets['train'], shuffle=True, batch_size=5)
    val_data_loaders = DataLoader(image_datasets['val'], shuffle=True, batch_size=5,)
    image_dataloaders = {'train': train_data_loaders, 'val': val_data_loaders}

    train_mean = torch.zeros(3)
    train_std = torch.zeros(3)
    val_mean = torch.zeros(3)
    val_std = torch.zeros(3)

    print('----training-----')
    for i, (inputs, labels) in enumerate(train_data_loaders):
        # print("inputs:", inputs.shape)
        # print("labels:", labels.shape)
        for i in range(3):
            train_mean[i] += inputs[:, i, :, :].mean()
            train_std[i] += inputs[:, i, :, :].std()

    train_mean = train_mean / len(image_datasets['train'])
    train_std = train_std / len(image_datasets['train'])
    print(train_mean, train_std)

    for inputs, labels in tqdm(val_data_loaders):
        for i in range(3):
            val_mean[i] += inputs[:, i, :, :].mean()
            val_std[i] += inputs[:, i, :, :].std()
    val_mean = val_mean / len(image_datasets['val'])
    val_std = val_std / len(image_datasets['val'])
    print(val_mean, val_std)

    return train_mean, train_std, val_mean, val_std



if __name__ == '__main__':
    consistent_mean, consistent_std = get_consistent_mean_and_std()
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(consistent_mean, consistent_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(consistent_mean, consistent_std)
        ]),
    }

    image_datasets = {x: CustomData(img_path='./split_data/' + x,
                                    data_transforms=data_transforms,
                                    dataset_split_name=x) for x in ['train', 'val']}
    train_data_loaders = DataLoader(image_datasets['train'], shuffle=True, batch_size=5,
                                    num_workers=1)
    val_data_loaders = DataLoader(image_datasets['val'], shuffle=True, batch_size=5,
                                  num_workers=1)
    image_dataloaders = {'train': train_data_loaders, 'val': val_data_loaders}

    #
    num_epochs = 100
    # model_ft = models.densenet121(pretrained=True)
    # model_ft.classifier = torch.nn.Linear(model_ft.classifier.in_features, 2)
    model_ft = models.densenet121(pretrained=False)
    num_final_feats = 10
    model_ft.classifier = torch.nn.Sequential(
        OrderedDict([
            ('fc1', torch.nn.Linear(model_ft.classifier.in_features, num_final_feats)),
            ('fc2', torch.nn.Linear(num_final_feats, 2))
    ]))

    # # resnet可以这样写
    # # model_ftrs = model_ft.fc.in_features
    # # model_ft.fc = nn.Linear(model_ftrs, 2)
    #
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = train_model(model_ft, image_dataloaders, criterion, optimizer, lr_scheduler, num_epochs,
                        use_gpu, image_datasets, model_name='densenet121_nopretrain')



