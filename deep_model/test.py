# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 2/25/2022 9:25 AM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
将train17中学习到的模型在整个数据集上进行微调，并在测试数据集上进行测试
"""
import os
import time
import pandas as pd
from torch import nn

try:
    from PIL import Image
except ImportError:
    print(ImportError)
    import Image

from tqdm import tqdm
from torchvision import datasets, models, transforms
import torch
from torch.utils.data import DataLoader, Dataset
from densenet_plus import Net2, Net4, Net8
from torch.optim import lr_scheduler
import numpy as np
from feat_tools import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
device_ids = [0,1,2,3]

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


def fine_tune(net, train_dataloader, img_datasets,
              num_epochs, criterion, optimizer, scheduler, use_gpu,
              model_dir='', model_name=" ", img_feats_dict=dict(), pretrained_file=''
              ):

    net.load_state_dict(torch.load(pretrained_file))

    best_acc = 0.0
    minimum_loss = 100000
    best_model_wts = net.state_dict()
    best_epoch = -1

    for epoch in range(num_epochs):

        net.train(True)
        # net.train(False)

        running_loss = 0.0
        running_corrects = 0.0

        preds_list = []
        labels_list = []

        for j, (paths, inputs, labels) in enumerate(train_dataloader):
            traditional_feats = []
            # for path in paths:
            #     if not path in img_feats_dict.keys():
            #         print('not matched')
            #     # print('res:', img_feats_dict[path])
            #     traditional_feats.append(img_feats_dict[path])
            traditional_feats = [img_feats_dict[path] for path in paths]
            traditional_feats = np.asarray(traditional_feats)
            traditional_feats = traditional_feats.astype(np.float64)
            traditional_feats = torch.Tensor(traditional_feats)
            # print('traditional feats shape:', traditional_feats.shape)

            if use_gpu:
                # print('yes gpu used')
                net = net.cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()
                traditional_feats = traditional_feats.cuda()

            optimizer.zero_grad()

            outputs = net.forward((inputs, traditional_feats))
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            # loss.backward()
            # optimizer.step()
            # scheduler.step()

            running_loss += loss.data
            # print("preds:", preds.data)
            preds_list.append(preds.data[0].item())
            # print("labels:", labels.data)
            labels_list.append(labels.data[0].item())
            running_corrects += torch.sum(preds == labels.data).to(torch.float32)

        train_loss = running_loss / len(img_datasets['train'])
        train_acc = running_corrects / len(img_datasets['train'])
        print('-' * 25, '*', '-' * 25)
        # print("PREDS:", preds_list)
        # print("LABELS:", labels_list)
        print('{} Loss:{:.4f} Acc:{:.4f}'.format('Train', train_loss, train_acc))

        # save_path = model_dir + '/' + \
        #             model_name + '_epoch_' + str(best_epoch) + '_' + str(best_acc) + '_best.pkl'
        # torch.save(net.state_dict(), save_path)

        if train_acc >= 0.99:
            print('reached 0.99')
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_acc = train_acc
            minimum_loss = train_loss
            break

        if train_acc >= best_acc and train_loss <= minimum_loss:
            best_epoch = epoch
            best_model_wts = net.state_dict()
            best_acc = train_acc
            minimum_loss = train_loss

        # if train_acc >= best_acc:
        #     best_epoch = epoch
        #     best_model_wts = net.state_dict()
        #     best_acc = train_acc

    save_path = model_dir + '/' + \
                model_name + '_epoch_' + str(best_epoch) + '_' + str(best_acc) + '_best.pkl'
    torch.save(best_model_wts, save_path)
    return save_path


def extra_feats_pipeline(img_data):
    # print("img_data type:", type(img_data))
    img_data = np.asarray(img_data)
    assert type(img_data) == np.ndarray
    mask_data = get_topo_mask(img_data)

    print('seg pixels:', np.sum(mask_data.astype(np.uint8) == 255))
    if np.sum(mask_data.astype(np.uint8) == 255) <= 10:
        return np.asarray([0] * 25)
    else:
        # 送入库函数来获取形状和纹理特征
        texture_feats = generate_single_texture_features(img_data, mask_data)

    filtered_feats = []
    # 找到最终选定特征中的所包含的拓扑特征的尺度
    topo_feats_keys = ['Degree', 'Vertex', 'Subgraph', 'Component', 'Average', 'Points', 'Diameter']
    # 根据之前筛选出来的特征关键词来获取有效特征
    feats_csv_file_1ist = ['../merged_features/filtered_features.csv',
                           '../merged_features/filtered_features_with_cv.csv',
                           '../merged_features/filtered_features_2.csv',
                           '../merged_features/filtered_features_3.csv',
                           ]
    file_idx = 0
    df = pd.read_csv(feats_csv_file_1ist[file_idx])
    df_columns = df.columns.tolist()[3:]
    scale_range_set = []
    for filtered_feat in df_columns:
        for key in topo_feats_keys:
            if key in filtered_feat:
                scale = filtered_feat.split('_')[-1]
                scale_range_set.append(scale)
                break
    scale_range_set = set(scale_range_set)
    # print('scale_range_set:', scale_range_set)
    scale_range_set = [int(i) for i in scale_range_set]

    # 送入拓扑特征提取函数获取拓扑特征
    mask_data = mask_data.astype(np.uint8)
    # print('mask data is:', mask_data)


    topo_feats = generate_single_topo_features(mask_data, scale_range_set)

    # print('--' * 10 + 'topo feats' + '--' * 10)
    # # print(topo_feats)
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
        filtered_feats.append(merged_feats[col])
    # print('final feats:', filtered_feats)
    # print(len(filtered_feats))
    filtered_feats = np.asarray(filtered_feats)

    return filtered_feats


@torch.no_grad()
def test_model(net, test_dataloader, img_datasets, use_gpu, img_feats_dict, save_path, phase='test'):

    net.load_state_dict(torch.load(save_path))

    net.train(False)
    # net.train(True)
    # net.eval()

    running_loss = 0.0
    running_corrects = 0.0

    preds_list = []
    labels_list = []
    paths_list = []

    for j, (paths, inputs, labels, extra_feats) in enumerate(test_dataloader):
        print('extra feats', extra_feats)
        print((extra_feats == torch.Tensor([0])).all().item())
        if (extra_feats == torch.Tensor([0])).all().item():
            print("skip this image with no calcification point")
            continue

        print("inputs shape:", inputs.shape)
        print("extra feats:", extra_feats.shape)

        # traditional_feats = []
        # traditional_feats = [img_feats_dict[path] for path in paths]
        # traditional_feats = np.asarray(traditional_feats)
        # traditional_feats = traditional_feats.astype(np.float64)
        # traditional_feats = torch.Tensor(traditional_feats)
        # print('traditional feats shape:', traditional_feats.shape)
        # print('TR Feats:', traditional_feats)
        # print('EX Feats:', extra_feats)

        if use_gpu:
            # print('yes gpu used')
            inputs = inputs.cuda()
            labels = labels.cuda()
            extra_feats = extra_feats.cuda()
            net = net.cuda()
            # traditional_feats = traditional_feats.cuda()

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
    test_loss = running_loss / len(img_datasets[phase])
    test_acc = running_corrects / len(img_datasets[phase])
    print('-' * 25, '*', '-' * 25)
    print("labels:", labels_list)
    print("preds:", preds_list)
    print("paths:", paths_list)
    print("correct num:", torch.sum(torch.Tensor(labels_list) == torch.Tensor(preds_list)).item())
    print("uncorrect num:", torch.sum(torch.Tensor(labels_list) != torch.Tensor(preds_list)).item())
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('Test:', test_loss, test_acc))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


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


def cmp_img(x, y, start_pos=0, end_pos=3):
    x_base = os.path.basename(x)
    y_base = os.path.basename(y)
    return int(x_base[start_pos:end_pos]) - int(y_base[start_pos:end_pos])


def get_consistent_mean_and_std(img_paths, img_labels):
    # img_paths = sorted(img_paths, key=cmp_to_key(cmp_img))
    # img_labels = [1 if 'M' in img_path else 0 for img_path in img_paths]

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


def get_img_feats_dict(feats_csv_file):
    df = pd.read_csv(feats_csv_file)
    img_feats_dict = dict()
    for index, row in df.iterrows():
        img_name = row['image'].replace('csv', 'bmp')
        img_feats = np.asarray(row[3:])
        img_feats_dict[img_name] = img_feats
    print('len of feats dict:', len(img_feats_dict))
    # print(img_feats_dict)
    return img_feats_dict


if __name__ == '__main__':
    # cur_img_path = '/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/resized_merged_test_data/roi/B17LMLO.jpg'
    # cur_img_path = '/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/resized_merged_test_data/roi/B11RCC.jpg'
    # cur_img_path = '/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/resized_merged_test_data/roi/M5RCC.jpg'
    #
    # cur_img_data = load_img_data_from_path(cur_img_path)
    # extra_feats = extra_feats_pipeline(cur_img_data)
    # extra_feats = torch.FloatTensor(extra_feats)
    #
    # try:
    #     os._exit(0)
    # except:
    #     print("Program Ends!")

    train_paths, train_labels = get_img_paths_and_labels(dir='../merged_data')
    print(len(train_labels))
    print(len(train_labels))
    # print(train_paths[:10])
    # print(train_labels[:10])
    test_paths, test_labels = get_img_paths_and_labels(dir='../resized_merged_test_data/roi')
    # test_paths, test_labels = get_img_paths_and_labels(dir='../merged_data')
    # print(test_paths[:5])
    # print(test_labels[:5])

    train_consistent_mean, train_consistent_std = get_consistent_mean_and_std(train_paths, train_labels)
    print(train_consistent_mean)
    print(train_consistent_std)

    test_consistent_mean, test_consistent_std = get_consistent_mean_and_std(test_paths, test_labels)
    # test_consistent_mean = train_consistent_mean
    # test_consistent_std = train_consistent_std

    print(test_consistent_mean)
    print(test_consistent_std)
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

    # train_data = TrainDataset(train_paths[:20], train_labels[:20], data_transforms['train'])
    train_data = TestDataset(train_paths[:20], train_labels[:20], data_transforms['train'])

    test_data = TestDataset(test_paths, test_labels, data_transforms['test'])
    img_datasets = {'train': train_data, 'test': test_data}

    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=1,
                                    num_workers=1)
    test_data_loaders = DataLoader(test_data, shuffle=True, batch_size=1,
                                  num_workers=1)
    dataloader = {'train': train_data_loaders, 'test': test_data_loaders}

    use_gpu = torch.cuda.is_available()
    
    # model_ft = models.densenet121(pretrained=True)
    # model = Net2(model_ft, num_mid_feats=256)
    # model_ft = models.densenet121(pretrained=True)
    # model = Net4(model_ft)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = 200
    weights = torch.FloatTensor([0.4, 0.6])

    feats_csv_file = '../merged_features/filtered_features.csv'
    img_feats_dict = get_img_feats_dict(feats_csv_file)
    print(img_feats_dict.keys())
    # print(type(img_feats_dict.keys()))
    # print('B100RCC.bmp' in img_feats_dict.keys())
    # print("feats_dict is:", img_feats_dict)
    # print(img_feats_dict['B38LMLO.bmp'])

    if use_gpu:
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
        weights = weights.cuda()
        criterion = nn.CrossEntropyLoss(weight=weights)

    # save_path = fine_tune(model, dataloader['train'], img_datasets, num_epochs, criterion, optimizer, scheduler,
    #                       use_gpu, model_dir='pkls_densenet121_2_cv_10_200', model_name="densenet121_2", img_feats_dict=img_feats_dict,
    #                       pretrained_file='pkls_densenet121_2_cv_10_200/0_densenet121_2_cv_10_epoch_54_tensor(0.9524, device=\'cuda:0\')_best.pkl')

    # save_path = fine_tune(model, dataloader['train'], img_datasets, num_epochs, criterion, optimizer, scheduler,
    #                       use_gpu, model_dir='pkls_densenet121_2_cv_10_200', model_name="densenet121_2", img_feats_dict=img_feats_dict,
    #                       pretrained_file='pkls_densenet121_2_cv_10_200/densenet121_2_epoch_157_tensor(1.0000, device=\'cuda:0\')_best.pkl')

    # save_path = fine_tune(model, dataloader['train'], img_datasets, num_epochs, criterion, optimizer, scheduler,
    #                       use_gpu, model_dir='pkls_densenet121_w_trfm_fusion_cv_10_200', model_name="densenet121_2", img_feats_dict=img_feats_dict,
    #                       pretrained_file='pkls_densenet121_w_trfm_fusion_cv_10_200/0_densenet121_w_trfm_fusion_cv_10_epoch_170_tensor(0.9048, device=\'cuda:0\')_best.pkl')

    # save_path = fine_tune(model, dataloader['train'], img_datasets, num_epochs, criterion, optimizer, scheduler,
    #                       use_gpu, model_dir='pkls_densenet121_w_trfm_fusion_cv_10_200', model_name="densenet121_2",
    #                       img_feats_dict=img_feats_dict,
    #                       pretrained_file='pkls_densenet121_w_trfm_fusion_cv_10_200/densenet121_2_epoch_11_tensor(0.9953, device=\'cuda:0\')_best.pkl')

    # save_path = 'pkls_densenet121_2_cv_10_200/densenet121_2_epoch_157_tensor(1.0000, device=\'cuda:0\')_best.pkl'
    # test_model(model, dataloader['test'], img_datasets, use_gpu, img_feats_dict, save_path)

    # test_model2(model, dataloader['train'], img_datasets, use_gpu, img_feats_dict, save_path)

    # save_path = '/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/deep_model2/pkls_densenet121_w_trfm_fusion_cv_10_200/' \
    #             'densenet121_2_epoch_0_tensor(0.9953, device=\'cuda:0\')_best.pkl'
    # test_model(model, dataloader['test'], img_datasets, use_gpu, img_feats_dict, save_path)


    save_path = 'pkls_densenet121_net8_cv_10_200/1_densenet121_net8_cv_10_epoch_21_tensor(0.9524, device=\'cuda:3\')_best.pkl'
    test_model(model, dataloader['test'], img_datasets, use_gpu, img_feats_dict, save_path)
