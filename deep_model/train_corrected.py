from collections import OrderedDict
import imp
from math import inf
from operator import mod
import random
from regex import P
from sklearn import metrics

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

import matplotlib.pyplot as plt


import shutil
from test_final import DatasetWithSingleImage, TestDatasetWithThresholdingData, get_consistent_mean_and_std

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


def setup_seed(seed):
    #  下面两个常规设置了，用来np和random的话要设置 
    np.random.seed(seed) 
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
    torch.manual_seed(seed)

    # torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


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


def train_model_corrected(
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
    # setup_seed(seed)

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
    # train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=3,  drop_last=True, generator=g,
    #                                 num_workers=1)
    # g.manual_seed(10)
    # val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=3,  drop_last=True, generator=g,
    #                         num_workers=1)

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=3, 
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=3,  
                            num_workers=1)

    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}


    best_epoch = -1
    best_val_acc = -1
    best_val_auc = -1
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

        print('**'*15, str(epoch) + "starting ", '**' * 15)

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            malignant_pred_probs_tensor = None

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
                    # topo_mask_data = topo_mask_data.cuda(device)
                    extra_feats = extra_feats.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                outputs = net.forward((inputs, extra_feats, ))

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                # outputs = net.forward((inputs, ))

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

                outputs_sf = F.softmax(outputs, -1)
                if j == 0:
                    malignant_pred_probs_tensor = outputs_sf[:, 1]
                else:
                    malignant_pred_probs_tensor = torch.cat((malignant_pred_probs_tensor, outputs_sf[:, 1]), 0)

            
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

            pred_probs_arr = malignant_pred_probs_tensor.cpu().detach().numpy()
            fpr, tpr, threshold = metrics.roc_curve(labels_arr, pred_probs_arr)

            correct_auc_1 = metrics.roc_auc_score(labels_arr, pred_probs_arr)

            correct_auc_2 = metrics.auc(fpr, tpr)

            assert correct_auc_1 == correct_auc_2
            
            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': pred_probs_arr}
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

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, correct_auc_1))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, correct_auc_1))
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
            best_val_auc = correct_auc_1

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)
            best_save_path = save_path

        else:
            count += 1
        #     if count >= 10:
        #         print("Training Process ends ")
        #         break


#################################################################################


    loss_and_acc_saved_dir = os.path.join(dir_path, 'lossAndacc')
    if not os.path.exists(loss_and_acc_saved_dir):
        os.mkdir(loss_and_acc_saved_dir)
    else:
        print(loss_and_acc_saved_dir, 'already exists!')

    train_acc_saved_path = os.path.join(loss_and_acc_saved_dir, model_name + '_train_acc.png')
    train_loss_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_train_loss.png')
    draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)

    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("Val auc:", best_val_auc)
    print("---------------------train end----------------------")
    # return save_path
    return best_save_path




@torch.no_grad()
def test_model_corrected(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1):

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

    malignant_pred_probs_tensor = None

    for j, (paths, inputs, labels, extra_feats, topo_mask_data) in enumerate(test_dataloader):

        if use_gpu:
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)

            extra_feats = extra_feats.cuda(device)

            net = net.cuda(device)

            # outputs = net.forward((inputs, ))
            outputs = net.forward((inputs, extra_feats,))


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

    dict = {'labels': labels_arr, 'preds': pred_probs_arr, }
    dict_filename = str(saved_epoch) + '_test_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts/'
    save_path = os.path.join(save_dir, dict_filename)

    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, correct_auc_1))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))




def train_model_w_loss_corrected(
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

        start_epoch = int(os.path.basename(pkl_file).split('_')[-3])

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
    best_val_auc = -1
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

        running_stages = ['train', 'val']

        print('**'*15, str(epoch) + "starting ", '**' * 15)

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
                # net.train(False)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            malignant_pred_probs_tensor = None

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                if use_gpu:
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
               
                    net = net.cuda(device)

                optimizer.zero_grad()

                outputs, outputs_1, outputs_2, weights = net.forward((inputs,))


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

                outputs_sf = F.softmax(outputs, -1)
                if j == 0:
                    malignant_pred_probs_tensor = outputs_sf[:, 1]
                else:
                    malignant_pred_probs_tensor = torch.cat((malignant_pred_probs_tensor, outputs_sf[:, 1]), 0)


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

            pred_probs_arr = malignant_pred_probs_tensor.cpu().detach().numpy()
            fpr, tpr, threshold = metrics.roc_curve(labels_arr, pred_probs_arr)

            correct_auc_1 = metrics.roc_auc_score(labels_arr, pred_probs_arr)

            correct_auc_2 = metrics.auc(fpr, tpr)

            assert correct_auc_1 == correct_auc_2

            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': pred_probs_arr}

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

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, correct_auc_1))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, correct_auc_2))
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
            best_val_auc = correct_auc_1

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)
            best_save_path = save_path

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
    draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)


    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("Val auc:", best_val_auc)
    print("---------------------train end----------------------")
    return best_save_path



def train_model_concat_fusion(
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

        start_epoch = int(os.path.basename(pkl_file).split('_')[-3])

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
    best_val_auc = -1
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

        running_stages = ['train', 'val']

        print('**'*15, str(epoch) + "starting ", '**' * 15)

        for phase in running_stages:

            if phase == 'train':
                net.train(True)
                # net.train(False)
            else:
                net.train(False)

            labels_list = []
            preds_list = []

            running_loss = 0.0

            malignant_pred_probs_tensor = None

            for j, (paths, inputs, labels, topo_mask_data) in enumerate(dataloader[phase]):

                if use_gpu:
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)

                    net = net.cuda(device)

                optimizer.zero_grad()

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                outputs, outputs_1, outputs_2 = net.forward((inputs,))


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

                outputs_sf = F.softmax(outputs, -1)
                if j == 0:
                    malignant_pred_probs_tensor = outputs_sf[:, 1]
                else:
                    malignant_pred_probs_tensor = torch.cat((malignant_pred_probs_tensor, outputs_sf[:, 1]), 0)


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

            pred_probs_arr = malignant_pred_probs_tensor.cpu().detach().numpy()
            fpr, tpr, threshold = metrics.roc_curve(labels_arr, pred_probs_arr)

            correct_auc_1 = metrics.roc_auc_score(labels_arr, pred_probs_arr)

            correct_auc_2 = metrics.auc(fpr, tpr)

            assert correct_auc_1 == correct_auc_2

            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': pred_probs_arr}

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

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, correct_auc_1))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, correct_auc_2))
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
            best_val_auc = correct_auc_1

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)
            best_save_path = save_path

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
    draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)


    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("Val auc:", best_val_auc)
    print("---------------------train end----------------------")
    return best_save_path



@torch.no_grad()
def test_model_w_loss_corrected(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1, w_loss=1.1):

    seed = 10
    setup_seed(seed)

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    print("mean ...",  test_consistent_mean)
    print("std ...", test_consistent_std)
    data_transforms = {
        phase: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms[phase])
    g = torch.Generator()
    g.manual_seed(seed)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=1, generator=g,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path, map_location=device))

    net.train(False)
    # net.train(True)

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
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            net = net.cuda(device)

        print("img:", paths)

        # if paths[0] == 'B27RCC.jpg':

        if paths == ('B27RCC.jpg',):

            print("inputs:", inputs)
       
        outputs, outputs_1, outputs_2, weights = net.forward((inputs,))

        _, preds = torch.max(outputs.data, 1)
        # preds = 1 - preds
        # preds_list.append(preds.data[0].item())
        # print("the preds is:", preds.data)
        # print("output:", outputs)
        for k_th in range(preds.shape[0]):
            print("the preds is:", preds[k_th].data)
            print("the labels is:", labels[k_th].data)

            preds_list.append(preds[k_th].item())
            labels_list.append(labels[k_th].item())
            paths_list.append(paths[k_th])

        print(')' * 20)

        loss = w_loss * criterion(outputs, labels) +  criterion(outputs_1, labels) +  criterion(outputs_2, labels)

        # print("the labels is:", labels.data)

        # labels_list.append(labels.data[0].item())

        # paths_list.append(paths)

        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

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
    dict_filename = str(saved_epoch) + '_test_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts/'
    save_path = os.path.join(save_dir, dict_filename)


    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, correct_auc_1))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))


    # 将分类错误的图片按照原本的标签 分成两个子文件夹进行保存
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/'
    original_save_dir = os.path.dirname(test_img_paths[0])
    wrong_pred_imgs_subdir = os.path.join(save_dir, 'proposed_wrong_pred_imgs')
    if not os.path.exists(wrong_pred_imgs_subdir):
        os.mkdir(wrong_pred_imgs_subdir)
    benign_mis_classified_subdir = os.path.join(wrong_pred_imgs_subdir, phase + '_benign_mis_classified')
    if not os.path.exists(benign_mis_classified_subdir):
        os.mkdir(benign_mis_classified_subdir)
    malignant_mis_classified_subdir = os.path.join(wrong_pred_imgs_subdir, phase + '_malignant_mis_classified')
    if not os.path.exists(malignant_mis_classified_subdir):
        os.mkdir(malignant_mis_classified_subdir)
    
    for benign_img_name in benign_mis_classified: #benign_img_name是只有一个元素的ｔｕｐｌｅ
        benign_img_original_path = os.path.join(original_save_dir, benign_img_name[0])
        benign_img_save_path = os.path.join(benign_mis_classified_subdir, benign_img_name[0])
        shutil.copy(benign_img_original_path, benign_img_save_path)

    for malignant_img_name in malignant_mis_classified:
        malignant_img_original_path = os.path.join(original_save_dir, malignant_img_name[0])
        malignant_img_save_path = os.path.join(malignant_mis_classified_subdir, malignant_img_name[0])
        shutil.copy(malignant_img_original_path, malignant_img_save_path)

    



    




@torch.no_grad()
def test_model_concat_fusion(net, test_img_paths, test_img_labels, use_gpu, save_path, criterion, phase='test', saved_epoch=-1, w_loss=1.1):

    seed = 10
    setup_seed(seed)

    print("testing model ...")
    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_img_paths, test_img_labels)
    print("mean ...",  test_consistent_mean)
    print("std ...", test_consistent_std)
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(test_consistent_mean, test_consistent_std)
        ]),
    }
    test_data = TestDatasetWithMaskData(test_img_paths, test_img_labels, data_transforms['test'])
    g = torch.Generator()
    g.manual_seed(seed)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=5, generator=g,
                                   num_workers=1)

    net.load_state_dict(torch.load(save_path, map_location=device))

    net.train(False)
    # net.train(True)

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
            inputs = inputs.cuda(device)
            labels = labels.cuda(device)
            net = net.cuda(device)

        print("img:", paths)

        # if paths[0] == 'B27RCC.jpg':

        if paths == ('B27RCC.jpg',):

            print("inputs:", inputs)
       
        outputs, outputs_1, outputs_2 = net.forward((inputs,))

        _, preds = torch.max(outputs.data, 1)
        # preds = 1 - preds
        # preds_list.append(preds.data[0].item())
        # print("the preds is:", preds.data)
        # print("output:", outputs)
        for k_th in range(preds.shape[0]):
            print("the preds is:", preds[k_th].data)
            print("the labels is:", labels[k_th].data)

            preds_list.append(preds[k_th].item())
            labels_list.append(labels[k_th].item())
            paths_list.append(paths[k_th])

        print(')' * 20)

        loss = w_loss * criterion(outputs, labels) +  criterion(outputs_1, labels) +  criterion(outputs_2, labels)

        # print("the labels is:", labels.data)

        # labels_list.append(labels.data[0].item())

        # paths_list.append(paths)

        print("corrects num:", torch.sum(preds == labels.data).to(torch.float32))

        running_loss += loss.data
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        if preds.data[0].item() == 1 and labels.data[0].item() == 0:
            benign_mis_classified.append(paths)
        if preds.data[0].item() == 0 and labels.data[0].item() == 1:
            malignant_mis_classified.append(paths)

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
    dict_filename = str(saved_epoch) + '_test_dicts.pickle'
    save_dir = os.path.dirname(save_path) + '/saved_dicts/'
    save_path = os.path.join(save_dir, dict_filename)


    with open(save_path, 'wb') as fp:
        pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


    auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

    print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format('Test:', test_loss, test_acc, correct_auc_1))
    print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
    print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))




def train_model_inception(
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

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=2, 
                                    num_workers=1)
    val_data_loaders = DataLoader(val_data, shuffle=True, batch_size=2,  
                            num_workers=1)

    dataloader = {'train': train_data_loaders, 'val': val_data_loaders}


    best_epoch = -1
    best_val_acc = -1
    best_train_acc = -1
    best_val_auc = -1
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

            malignant_pred_probs_tensor = None

            for j, (paths, inputs, labels) in enumerate(dataloader[phase]):


                if use_gpu:
                    # print('yes gpu used')
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    net = net.cuda(device)

                optimizer.zero_grad()

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

                outputs_sf = F.softmax(outputs1, -1)
                if j == 0:
                    malignant_pred_probs_tensor = outputs_sf[:, 1]
                else:
                    malignant_pred_probs_tensor = torch.cat((malignant_pred_probs_tensor, outputs_sf[:, 1]), 0)


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

            pred_probs_arr = malignant_pred_probs_tensor.cpu().detach().numpy()
            fpr, tpr, threshold = metrics.roc_curve(labels_arr, pred_probs_arr)

            correct_auc_1 = metrics.roc_auc_score(labels_arr, pred_probs_arr)

            correct_auc_2 = metrics.auc(fpr, tpr)

            assert correct_auc_1 == correct_auc_2
            
            auc_score, sensitivity, specificity, ppv, npv = cal_metrics(labels_arr, preds_arr)

            dict = {'labels': labels_arr, 'preds': pred_probs_arr}
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

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, train_loss, train_acc, correct_auc_1))
                print('Sensitivity:{:.4f} Specificity:{:.4f}'.format(sensitivity, specificity))
                print('ppv:{:.4f} npv:{:.4f}'.format(ppv, npv))

                print('-'*25)

            else:
                val_loss = running_loss / len(preds_list)
                val_acc = correct_num / len(preds_list)

                print('{} Loss:{:.4f} Acc:{:.4f} Auc:{:.4f}'.format(phase, val_loss, val_acc, correct_auc_2))
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
            best_val_auc = correct_auc_1

            save_path = dir_path + '/' + \
            str(model_name) + '_epoch_' + str(epoch) + '_' + str(train_acc) + '_'  + str(val_acc) + '.pkl'
            torch.save(net.state_dict(), save_path)
            best_save_path = save_path

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
    draw_fig(train_acc_list, 'acc', num_epochs, train_acc_saved_path)
    draw_fig(train_loss_list, "loss", num_epochs, train_loss_saved_path)


    val_acc_saved_path = os.path.join(loss_and_acc_saved_dir,  model_name + '_val_acc.png')
    val_loss_saved_path = os.path.join(loss_and_acc_saved_dir,   model_name + '_val_loss.png')
    draw_fig(val_acc_list, 'acc', num_epochs, val_acc_saved_path)
    draw_fig(val_loss_list, 'loss', num_epochs, val_loss_saved_path)


    print("---------------------train start----------------------")
    print("Best epoch:", best_epoch)
    print("Val Loss:", best_val_loss)
    print("Val Acc:", best_val_acc)
    print("Val Auc:", best_val_auc)
    print("---------------------train end----------------------")
    return best_save_path
