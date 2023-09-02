# -*- encoding: utf-8 -*-
"""
@File    : sklearn_test.py
@Time    : 3/8/2022 11:10 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import os
import sys
import cv2
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import numpy as np
from feat_tools import *
import pandas as pd
from __init__ import global_var
import pickle


def extra_feats_pipeline(img_data, mean_feats_dict, std_feats_dict):
    # print("img_data type:", type(img_data))
    img_data = np.asarray(img_data)
    assert type(img_data) == np.ndarray
    
    mask_arr = np.ones((256, 256), dtype=np.uint8) * 255
    mask_data = cv2.cvtColor(mask_arr, cv2.COLOR_GRAY2RGB)
   
    # 送入库函数来获取形状和纹理特征
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
    scale_range_set = []

    for filtered_feat in df_columns:
        for key in topo_feats_keys:
            if key in filtered_feat and 'SumAverage' not in filtered_feat:
                print("key:", key)
                print(filtered_feat)
                scale = filtered_feat.split('_')[-1]
                scale_range_set.append(scale)
                break
    scale_range_set = set(scale_range_set)
    print('scale_range_set:', scale_range_set)
    scale_range_set = [int(i) for i in scale_range_set]

    # 送入拓扑特征提取函数获取拓扑特征
    topo_feats = []
    threshold = 20
    while topo_feats == []:
        topo_mask_data = get_topo_mask(img_data, threshold=threshold)
        # 送入拓扑特征提取函数获取拓扑特征
        topo_mask_data = topo_mask_data.astype(np.uint8)
        topo_feats = generate_single_topo_features(topo_mask_data, scale_range_set)
        threshold -= 1

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

    return filtered_feats



if __name__ == '__main__':

    # base_dir = global_var.base_test_data_prefix + '/roi'
    base_dir = global_var.base_data_prefix + '/roi'

    # model = joblib.load(global_var.base_sklearn_prefix + '/model_saved/svc_on_whole_25.pkl')
    # model = joblib.load(global_var.base_sklearn_prefix + '/model_saved/svc_on_whole_cv.pkl')
    # model = joblib.load(global_var.base_sklearn_prefix + '/model_saved/svc_on_whole_10.pkl')
    # model = joblib.load(global_var.base_sklearn_prefix + '/model_saved/svc_on_whole_20.pkl')
    # model = joblib.load(global_var.base_sklearn_prefix + '/model_saved/svc_on_whole_30.pkl')
    model = joblib.load(global_var.base_sklearn_prefix + '/model_saved/svc_on_whole_10.pkl')

    
    preds = []
    labels = []

    with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict2.pkl', "rb") as f:
        mean_feats_dict = pickle.load(f)
    with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict2.pkl', "rb") as f:
        std_feats_dict = pickle.load(f)


    for img_name in os.listdir(base_dir):
        # if img_name != 'B48LCC.jpg':
        #     continue

        label = 0 if img_name[0] == 'B' else 1
        img_path = os.path.join(base_dir, img_name)
        cur_img_data = cv2.imread(img_path)

        extra_feats = extra_feats_pipeline(cur_img_data, mean_feats_dict, std_feats_dict)

        if (torch.Tensor(extra_feats) == torch.Tensor([0])).all().item():
            print('No extra feats')
            extra_feats = torch.Tensor(extra_feats)
            continue
        else:
            print("Extra Feats!")
            extra_feats = np.asarray(extra_feats)
            print("extra_feats:", extra_feats)

            extra_feats = torch.Tensor(extra_feats)

            extra_feats = extra_feats.reshape(1, -1)
            # cancer_degree_probility_arr = model.predict_proba(extra_feats)[0]
            # print('cancer_degree_probility_arr', cancer_degree_probility_arr)
            # cancer_degree = np.argmax(cancer_degree_probility_arr)
            # print('cancer degree:', cancer_degree)
            # cancer_degree_prob = cancer_degree_probility_arr[cancer_degree]
            # print('cancer degree_prob:', cancer_degree_prob)
            cancer_degree = model.predict(extra_feats)[0]

            print("label:", label)
            print("pred:", cancer_degree)
            labels.append(label)
            preds.append(cancer_degree)

    print("preds:", preds)
    print("labels:", labels)
    corrects = torch.sum(torch.Tensor(preds) == torch.Tensor(labels)).item()
    uncorrects = torch.sum(torch.Tensor(preds) != torch.Tensor(labels)).item()
    print("correct num:", corrects)
    print("uncorrect num:", uncorrects)
    print("acc:", corrects / (corrects + uncorrects) )




