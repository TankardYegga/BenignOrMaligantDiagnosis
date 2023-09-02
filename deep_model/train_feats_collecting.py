import csv
import os

import torch
from train29 import get_consistent_mean
from train29 import get_available_data_by_order
from test_final import DatasetWithMaskData, TestDataset
from __init__ import global_var
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd


def collect_feats(train_dir, save_path):
    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)
    # train_img_paths = train_img_paths[:5]
    # train_img_labels = train_img_labels[:5]

    train_consistent_mean, train_consistent_std = get_consistent_mean(train_img_paths, train_img_labels)
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = TestDataset(train_img_paths, train_img_labels, data_transforms['train'])
    batch_size = 5
    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=batch_size,
                                        num_workers=1)


    feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_with_cv.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_20.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_30.csv']
    # feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_9.csv']

    file_idx = 0
    df = pd.read_csv(feats_csv_file_1ist[file_idx])
    df_columns = df.columns.tolist()[3:]

    header = ['image', 'label']
    header += df_columns

    if not os.path.exists(save_path):
        img_lists = []
        with open(save_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    else:
        df = pd.read_csv(save_path)
        img_lists = df['image'].tolist()

    csv_content_rows = []

    for j, (paths, inputs, labels, extra_feats) in enumerate(train_data_loaders):

        # if j == 1:
        #     break
        
        print('9' * 30)
        print("shape:", extra_feats.shape)
        print(extra_feats)
        print('9' * 30)

        for i in range(len(paths)):

            csv_content_row = []
            path = paths[i]

            if path in img_lists:
                print('--------------skipped------------')
                print(path)
                print('--------------skipped------------')
                continue

            label = labels[i]
            print('type', type(extra_feats[i]))
            extra_feat = (extra_feats[i]).tolist()
            csv_content_row.append(path)
            csv_content_row.append(label.item())
            csv_content_row += extra_feat
            csv_content_rows.append(csv_content_row)
            with open(save_path, "a", newline='') as f:
                writer = csv.writer(f)    
                writer.writerow(csv_content_row)
          
        print("10" * 30)
    
    # with open(save_path, "w", newline='') as f:
    #     writer = csv.writer(f)    
    #     for csv_content_row in csv_content_rows:
    #         writer.writerow(csv_content_row)


def test_collect_feats(train_dir, save_path):
    train_img_paths, train_img_labels = get_available_data_by_order(data_dir=train_dir)

    # train_consistent_mean, train_consistent_std = get_consistent_mean(train_img_paths, train_img_labels)
    train_consistent_mean = torch.tensor([0.4330, 0.4330, 0.4330])
    train_consistent_std = torch.tensor([0.1261, 0.1261, 0.1261])

    train_img_paths = train_img_paths[1:2]
    train_img_labels = train_img_labels[1:2]

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_consistent_mean, train_consistent_std)
        ]),
    }

    train_data = TestDataset(train_img_paths, train_img_labels, data_transforms['train'])
    batch_size = 1
    train_data_loaders = DataLoader(train_data, shuffle=True, batch_size=batch_size,
                                        num_workers=1)


    feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv']

    file_idx = 0
    df = pd.read_csv(feats_csv_file_1ist[file_idx])
    df_columns = df.columns.tolist()[3:]

    header = ['image', 'label']
    header += df_columns

    for j, (paths, inputs, labels, extra_feats) in enumerate(train_data_loaders):
        
        print('9' * 30)
        print("paths:", paths)
        print("shape:", extra_feats.shape)
        print(extra_feats)
        print('9' * 30)

      
 
if __name__ == "__main__":

    # save_path = global_var.base_data_aug_4_prefix  + '/filtered_features_10.csv'
    # train_dir = global_var.base_data_aug_4_prefix + '/roi'
    # collect_feats(train_dir, save_path)

    # test_collect_feats(train_dir, save_path)


    # save_path = global_var.base_data_trains256_prefix  + '/filtered_features_10.csv'
    # train_dir = global_var.base_data_trains256_prefix + '/roi'
    # collect_feats(train_dir, save_path)

    # save_path =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_1/val/filtered_features_10.csv'
    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_1/val/roi'
    # collect_feats(train_dir, save_path)

    # save_path =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_2/val/filtered_features_10.csv'
    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_2/val/roi'
    # collect_feats(train_dir, save_path)

    # save_path =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_2/train/filtered_features_10.csv'
    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_2/train/roi'
    # collect_feats(train_dir, save_path)


    # save_path =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/val/filtered_features_10.csv'
    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/val/roi'
    # collect_feats(train_dir, save_path)

    # save_path =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/train/filtered_features_10.csv'
    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/train/roi'
    # collect_feats(train_dir, save_path)

    # save_path =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/filtered_features_10.csv'
    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    # collect_feats(train_dir, save_path)


    # save_path =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/filtered_features_10.csv'
    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/roi'
    # collect_feats(train_dir, save_path)


    # save_path =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/train/filtered_features_10.csv'
    # train_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/train/roi'
    # collect_feats(train_dir, save_path)


   

