# -*- encoding: utf-8 -*-
"""
@File    : train6.py.py
@Time    : 12/21/2021 3:36 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import time
import os
from dataset import CustomData
from densenet2 import DenseNet


def train_model(model, dataloader, criterion, optimizer, scheduler,
                num_epochs, use_gpu, img_datasets, model_name='manual_densenet', pkl_file=''):

    start_epoch = 0
    print('pickle file is:', pkl_file)
    print(os.path.exists(pkl_file))
    if os.path.exists(pkl_file):
        model.load_state_dict(torch.load(pkl_file))
        # start_epoch = os.path.basename(pkl_file).split('_')[-1].split('.')[0]
        start_epoch = int(os.path.basename(pkl_file).split('_')[-2])
        print('the start epoch is:', start_epoch)
    else:
        print('the start epoch is:', start_epoch)
        # return

    model_pkls_name = 'pkls_' + model_name + '_' + str(num_epochs)
    print("model_pkls_name:", model_pkls_name)
    dir_path = "deep_model2/" + model_pkls_name
    print("dir:", dir_path)
    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)
            print('make succ:', dir_path)
        except Exception as e:
            print(e)

    best_acc = 0.0
    best_model_wts = model.state_dict()
    best_epoch = -1
    since = time.time()

    for epoch in range(start_epoch + 1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 20)

        # if epoch % 5 == 0:
        #     running_stages = ['train', 'val']
        # else:
        #     running_stages = ['train']

        running_stages = ['train', 'val']

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

                # preds_one_hot = torch.eye(2)[preds]
                # preds_one_hot = Variable(preds_one_hot, requires_grad=True)
                # labels_one_hot = torch.eye(2)[labels]
                # loss = criterion(preds_one_hot, labels_one_hot)
                # print('loss is', loss)
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

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects / len(image_datasets[phase])

            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                # print('train yes')
                save_path = './deep_model2/' + model_pkls_name + '/' + \
                           str(model_name) + '_epoch_' + str(epoch) + '.pkl'
                # print('save path:', save_path)
                torch.save(model.state_dict(), save_path)

            if phase == 'val' and epoch_acc > best_acc:
                best_model_wts = model.state_dict()
                best_acc = epoch_acc
                best_epoch = epoch

    elasped_time = time.time() - since
    print('Training and Validating finished in: {:.0f} m {:.0f}s'.format(
        elasped_time // 60, elasped_time % 60
    ))
    print('Best Acc:{:.4f}'.format(best_acc))
    print('Best Epoch:', best_epoch)
    torch.save(best_model_wts, './deep_model2/'  + model_pkls_name + '/' + str(model_name) \
               + '_epoch_' + str(best_epoch)  + '_' + str(best_acc) + '_best.pkl')


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
    train_data_loaders = DataLoader(image_datasets['train'], shuffle=True, batch_size=10,
                                    num_workers=1)
    val_data_loaders = DataLoader(image_datasets['val'], shuffle=True, batch_size=10,
                                  num_workers=1)
    image_dataloaders = {'train': train_data_loaders, 'val': val_data_loaders}

    train_mean = torch.zeros(3)
    train_std = torch.zeros(3)
    val_mean = torch.zeros(3)
    val_std = torch.zeros(3)

    print('----training-----')
    for inputs, labels in tqdm(train_data_loaders):
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


def get_consistent_mean_and_std():
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
    train_data_loaders = DataLoader(image_datasets['train'], shuffle=True, batch_size=10,
                                    num_workers=1)
    val_data_loaders = DataLoader(image_datasets['val'], shuffle=True, batch_size=10,
                                  num_workers=1)

    train_mean = torch.zeros(3)
    train_std = torch.zeros(3)

    print('----training-----')
    for inputs, labels in tqdm(train_data_loaders):
        # print("inputs:", inputs.shape)
        # print("labels:", labels.shape)
        for i in range(3):
            train_mean[i] += inputs[:, i, :, :].mean()
            train_std[i] += inputs[:, i, :, :].std()

    for inputs, labels in tqdm(val_data_loaders):
        for i in range(3):
            train_mean[i] += inputs[:, i, :, :].mean()
            train_std[i] += inputs[:, i, :, :].std()

    train_mean = train_mean / (len(image_datasets['train']) + len(image_datasets['val']) )
    train_std = train_std / (len(image_datasets['train']) + len(image_datasets['val']) )
    print(train_mean, train_std)

    return train_mean, train_std


if __name__ == '__main__':
    # train_mean, train_std, val_mean, val_std = get_normalized_mean_and_std()
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(train_mean, train_std)
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(val_mean, val_std)
    #     ]),
    # }

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
    num_epochs = 150
    model_ft = DenseNet(num_final_feats=25)

    # # resnet可以这样写
    # # model_ftrs = model_ft.fc.in_features
    # # model_ft.fc = nn.Linear(model_ftrs, 2)
    #
    use_gpu = torch.cuda.is_available()
    # weights = torch.FloatTensor([0.3, 0.7])
    weights = torch.FloatTensor([0.4, 0.6])
    # weights = torch.FloatTensor([0.35, 0.65])
    if use_gpu:
        model_ft = model_ft.cuda()
        weights = weights.cuda()

    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # model = train_model(model_ft, image_dataloaders, criterion, optimizer, lr_scheduler, num_epochs,
    #                     use_gpu, image_datasets,
    #                     pkl_file=r'./deep_model2/pkls_manual_densenet_100_1/manual_densenet_epoch_76_best.pkl')
    model = train_model(model_ft, image_dataloaders, criterion, optimizer, lr_scheduler, num_epochs,
                        use_gpu, image_datasets, model_name='manual_densenet_15',
                        pkl_file=r'')
