# -*- encoding: utf-8 -*-
"""
@File    : generate_topo_mask.py
@Time    : 3/7/2022 4:12 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import os

import cv2
import networkx
import pywt
import numpy as np
import matplotlib.pyplot as plt
from radiomics import featureextractor, shape2D
import SimpleITK as sitk
from time import time
from __init__ import global_var

import pandas as pd

import sys
sys.path.append('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/deep_model/')

from feat_tools import generate_single_topo_features


def get_topo_mask(gray_img_arr, threshold = 20):
    if len(gray_img_arr.shape) == 3:
        gray_img_arr = cv2.cvtColor(gray_img_arr, cv2.COLOR_RGB2GRAY)
    assert len(gray_img_arr.shape) == 2

    coeffs = pywt.wavedec2(gray_img_arr, 'haar', level=3)
    cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    reconstructed_img = pywt.waverec2(coeffs, 'haar')
    coeffs[0] = np.zeros_like(coeffs[0])
    coeffs_2 = list(coeffs[-2])
    coeffs_1 = list(coeffs[-1])
    coeffs_2[-1] = np.zeros_like(coeffs_2[-1])
    coeffs_1[-1] = np.zeros_like(coeffs_1[-1])
    coeffs[-1] = tuple(coeffs_1)
    coeffs[-2] = tuple(coeffs_2)
    reconstructed_img = pywt.waverec2(coeffs, 'haar')
    another_img = np.zeros_like(reconstructed_img)
    for i in range(reconstructed_img.shape[0]):
        for j in range(reconstructed_img.shape[-1]):
            if reconstructed_img[i][j] <= -1 * threshold:
                another_img[i][j] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    topo_mask_img = cv2.morphologyEx(another_img, cv2.MORPH_CLOSE, kernel)
    # print(topo_mask_img.shape)
    # print(np.min(topo_mask_img))
    # print(np.max(topo_mask_img))
    # cv2.imwrite('topo.png', topo_mask_img)
    topo_mask_img = np.float32(topo_mask_img)
    topo_mask_img = cv2.cvtColor(topo_mask_img, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('topo_color.png', topo_mask_img)
    return topo_mask_img


# 自适应调整threshold的值
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
    
    return np.asarray(topo_mask_data)


def app1():
    # roi_dir = 'D:\\AIImageProject\\2DEMO\\BreastCancerDiagnosis\\algorithm\\Data\\roi'
    # save_dir = 'D:\\AIImageProject\\2DEMO\\BreastCancerDiagnosis\\algorithm\\Data\\topo_mask'

    # roi_dir = global_var.base_data_prefix + '/no_voxels_roi'
    # save_dir = global_var.base_data_prefix + '/no_voxels_mask'
    
    # for roi_img in os.listdir(roi_dir):
    #     roi_img_name = os.path.basename(roi_img).split('.')[0]

    #     roi_img_path = os.path.join(roi_dir, roi_img)
    #     roi_img_arr = cv2.imread(roi_img_path)
    #     topo_mask_arr = get_topo_mask(roi_img_arr, threshold=8)

    #     cv2.imwrite(os.path.join(save_dir, roi_img_name + '_mask.jpg'), topo_mask_arr)
    pass


def app2():
    base_dir = global_var.base_data_prefix + '/roi'
    save_dir = global_var.base_data_prefix + '/topo_mask'
    img_lists = ['B48LMLO.jpg', 'B25RCC.jpg', 'M33LCC.jpg', 'B51LMLO.jpg']

    for img in img_lists:
        img_name = img.split('.')[0]

        img_path = os.path.join(base_dir, img)
        img_arr = cv2.imread(img_path)
        topo_mask_arr = get_topo_mask(img_arr, threshold=8)

        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app3():
    roi_dir = global_var.base_data_aug_4_prefix + '/roi'
    save_dir = global_var.base_data_aug_4_prefix + '/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app4():
    roi_dir = global_var.base_data_aug_2_prefix + '/roi'
    save_dir = global_var.base_data_aug_2_prefix + '/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app5():

    roi_dir = global_var.base_data_aug_3_prefix + '/roi'
    save_dir = global_var.base_data_aug_3_prefix + '/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app6():

    roi_dir = global_var.base_data_aug_9_prefix + '/roi'
    save_dir = global_var.base_data_aug_9_prefix + '/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app7():

    roi_dir = global_var.base_data_aug_prefix + '/roi'
    save_dir = global_var.base_data_aug_prefix + '/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app8():

    roi_dir = global_var.base_data_trains256_prefix + '/trains_256/roi'
    save_dir = global_var.base_data_trains256_prefix + '/trains_256/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app9():

    roi_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/roi'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app10():

    roi_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app11():

    roi_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/roi'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2/topo_mask'

    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app12():

    roi_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_1/train/roi'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_1/train/topo_mask'


    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)



def app13():

    roi_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_2/train/roi'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_2/train/topo_mask'


    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)



def app14():

    roi_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/train/roi'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/train/topo_mask'


    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app15():

    roi_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/roi'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/topo_mask'


    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)


def app16():

    roi_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test2/roi'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test2/topo_mask'


    for img in os.listdir(roi_dir):
        img_name = img.split('.')[0]

        img_path = os.path.join(roi_dir, img)

        img_arr = cv2.imread(img_path)
        topo_mask_arr = topo_mask_data_pipeline(img_arr)
        
        cv2.imwrite(os.path.join(save_dir, img_name + '_mask.jpg'), topo_mask_arr)

if __name__ == '__main__':

    # app3()

    # app4()

    # app5()

    # app6()

    # app7()

    # app8()

    # app11()

    # app12()

    # app13()

    # app14()

    # app15()

    app16()

   








