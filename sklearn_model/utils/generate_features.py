# -*- encoding: utf-8 -*-
"""
@File    : generate_features.py
@Time    : 11/3/2021 10:18 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import cv2
import csv
import SimpleITK as sitk
from radiomics import featureextractor, shape2D
import os
import numpy as np

import sys
import path
sys.path.append('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets_api')

from generate_mask import process_all_imgs,find_all_required_files
from topology_extractor import feature_extractor_main

from __init__ import global_var

def generate_single_topo_features(case_img_path):
    return feature_extractor_main(case_img_path)

def get_series_number(mask_path):
    """
    series number >= 0
    """
    if mask_path[3].isdigit():
        return int(mask_path[1:4])
    elif mask_path[2].isdigit():
        return int(mask_path[1:3])
    elif mask_path[1].isdigit():
        return int(mask_path[1:2])
    else:
        return -1

def generate_topology_features(mask_path, save_path):
    benign_features = []
    malignant_features = []

    # 排序，顺序是B1 B2 B3 ... B120  M1 M2 .. M120
    mask_path_lists = os.listdir(mask_path)
    mask_path_dict = {mask_path:get_series_number(mask_path) for mask_path in mask_path_lists}
    mask_path_lists = sorted(mask_path_dict.items(), key=lambda x: x[1])
    mask_path_lists = sorted(dict(mask_path_lists).items(), key=lambda x:x[0][0])
    mask_path_lists = list(dict(mask_path_lists).keys())
    print('mask lists:', mask_path_lists)

    for case_mask_img in mask_path_lists:
        # skipped_img_lists = ['B12RMLO_mask.jpg', 'B34RCC_mask.jpg', 'B35LMLO_mask.jpg',
        # 'B47LMLO_mask.jpg','B52LMLO_mask.jpg', 'B62LCC_mask.jpg', 'B62LMLO_mask.jpg', 
        # 'B68RCC_mask.jpg', 'B68RMLO_mask.jpg']
        # if os.path.basename(case_mask_img) in skipped_img_lists:
        #     print('skipped')
        #     continue
      
        img_name = os.path.basename(case_mask_img)[:-9] + ".jpg"
        if os.path.exists(save_path + img_name.replace("jpg", "csv")):
            print("Existed!")
            continue
        print("mask %s" % case_mask_img)

        # skipped_imgs = ['B52LMLO_mask.jpg', 'B62LCC_mask.jpg', 'B62LMLO_mask.jpg']
        # if case_mask_img in skipped_imgs:
        #     continue
        # series_num = case_mask_img[1:4] if case_mask_img[3].isdigit() else case_mask_img[1:3]
        # print('series num:', series_num)
        # if int(series_num) <= 52:
        #     continue

        case_img_path = os.path.join(mask_path, case_mask_img)

        topofs = generate_single_topo_features(case_img_path)

        keys, values = [], []
        for key, value in topofs.items():
            keys.append(key)
            values.append(value)

        # 去掉名字中的_mask
        case_mask_img = case_mask_img.split('_')[0] + '.jpg'
        with open(save_path + case_mask_img.replace("jpg", "csv"), "w", newline='') as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(keys)
            csvwriter.writerow(values)


def generate_single_texture_features(img_path, mask_path, save_path):

    img_name = os.path.basename(img_path)

    if os.path.exists(save_path + img_name.replace("jpg", "csv")):
        print("already calculated texture features")
        return 

    img2d_arr = cv2.imread(img_path)
    print(np.max(img2d_arr))
    mask2d_arr = cv2.imread(mask_path)
    print(mask2d_arr.shape)
    print(img2d_arr.shape)
    assert img2d_arr.shape == mask2d_arr.shape

    print("min 0:", np.min(mask2d_arr))
    print("max 0:", np.max(mask2d_arr))
    if(np.count_nonzero(mask2d_arr == 255 ) <= 10):
        print("cali points too few!")
        print("cal points num:", np.count_nonzero(mask2d_arr == 255 ))
        return

    img2d_gray_arr = cv2.cvtColor(img2d_arr, cv2.COLOR_BGR2GRAY)
    mask2d_gray_arr = cv2.cvtColor(mask2d_arr, cv2.COLOR_BGR2GRAY)
    print("image2d shape:", img2d_gray_arr.shape)
    print("mask2d shape:", type(mask2d_gray_arr))
    print("min 1:", np.min(mask2d_gray_arr))
    print("max 1:", np.max(mask2d_gray_arr))
    image2d = sitk.GetImageFromArray(img2d_gray_arr)
    mask2d = sitk.GetImageFromArray(mask2d_gray_arr)
  

    mask2d_trans = sitk.GetArrayFromImage(mask2d)
    print("min:", np.min(mask2d_trans))
    print("max:", np.max(mask2d_trans))
    print("equal:", np.all(mask2d_trans == mask2d_gray_arr))

    settings = {
        'binWidth': 20,
        'sigma': [1, 2, 3],
        'verbose': True,
        # 'force2D': True,
        'label': 255
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(additionInfo=True, **settings)
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('LBP2D')
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableFeatureClassByName('shape2D')
    result = extractor.execute(image2d, mask2d)
    print("result length: ", len(result))

    pop_keys = list(result.keys())[:22]
    for key in list(result.keys()):
        if key in pop_keys:
            result.pop(key)

    supplemented_result = dict()
    settings2 = {
        'force2D': True,
        'label': 255
    }
    shape2DFeatures = shape2D.RadiomicsShape2D(image2d, mask2d, **settings2)
    shape2DFeatures.enableAllFeatures()
    shape2DFeatures.execute()
    supplemented_result.update({'original_shape2D_SphericalDisproportion':
                                    shape2DFeatures.getSphericalDisproportionFeatureValue()})
    for key, val in result.items():
        supplemented_result.update({key: val})
    assert len(supplemented_result) == 847

    keys, values = [], []
    for key, value in supplemented_result.items():
        keys.append(key)
        values.append(value)

    with open(save_path + img_name.replace("jpg", "csv"), "w", newline='') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(keys)
        csvwriter.writerow(values)


def generate_texture_features(img_base_dir, mask_base_dir):
    """
    :param base_dir:
    :param label_info_idx: 表示图像标签的关键字在路径中的序号，从0开始计数
    :return:
    """
    file_extensions_list = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG', '.bmp']
    img_file_path_list = []
    mask_file_path_list = []
    find_all_required_files(file_extensions_list, img_base_dir, img_file_path_list)
    find_all_required_files(file_extensions_list, mask_base_dir, mask_file_path_list)
    img_file_path_list.sort()
    mask_file_path_list.sort()
    print(len(img_file_path_list))
    print(img_file_path_list)
    print(len(mask_file_path_list))
    print(mask_file_path_list)

    # for file_idx in range(0, len(file_path_list), 2):
    #     img_path = file_path_list[file_idx]
    #     mask_path = file_path_list[file_idx + 1]
    #     img_label = img_path.split("\\")[label_info_idx]
    #     print("img path is:", img_path)
    #     print("mask path is:", mask_path)
    #     generate_single_texture_features(img_path, mask_path, img_label)
    for img_file_path in img_file_path_list:
        img_name = os.path.basename(img_file_path).split('.')[0]
        mask_file_path = os.path.join(mask_base_dir, img_name + "_mask.jpg")

        if mask_file_path not in mask_file_path_list:
            print(img_name, 'not matched')
            print(mask_file_path)
            # continue
        print('mask file:', mask_file_path)
        generate_single_texture_features(img_file_path, mask_file_path, 
         save_path = global_var.base_sklearn_prefix + "/features_saved/texture_features/")


def merge_features(texture_feat_dir, topo_feat_dir, merged_feat_save_path):
    """
    :param texture_feat_dir:
    :param topo_feat_dir:
    :param merged_feat_save_path:
    :return:
    """
    """将拓扑特征和纹理特征全部放在一个文件里面"""
    def merge_csv_header(texture_csv_example, topo_csv_example, filtered_header = ['image', 'label']):
        merged_csv_header = ['image', 'label']

        with open(texture_csv_example, "r") as f1:
            reader1 = csv.reader(f1)
            reader1_list = list(reader1)
            header1 = reader1_list[0]
            header1_copy = header1.copy()
            for key in header1:
                if key in filtered_header:
                    header1_copy.remove(key)
            merged_csv_header += header1_copy

        with open(topo_csv_example, "r") as f2:
            reader2 = csv.reader(f2)
            reader2_list = list(reader2)
            header2 = reader2_list[0]
            header2_copy = header2.copy()
            for key in header2:
                if key in filtered_header:
                    header2_copy.remove(key)
            merged_csv_header += header2_copy

        return merged_csv_header

    def merge_csv_content(texture_feat_dir, topo_feat_dir, filtered_header = ['image', 'label']):
        merged_csv_content = []
        for csv_file in os.listdir(texture_feat_dir):
            case_complete_features = []
            # separator_idx = csv_file.index('_')
            # csv_file_image = csv_file[0] + csv_file[separator_idx+1:-4] + '.jpg'
            # csv_file_label = csv_file[:separator_idx]
            csv_file_image = csv_file
            csv_file_label = 'Benign' if csv_file[0] == 'B' else 'Malignant'
            case_complete_features.append(csv_file_image)
            case_complete_features.append(csv_file_label)

            with open(os.path.join(texture_feat_dir, csv_file), "r") as f1:
                reader1_list = list(csv.reader(f1))
                header1 = reader1_list[0]
                content1 = reader1_list[1]
                for i in range(len(header1)):
                    if header1[i] not in filtered_header:
                        case_complete_features.append(content1[i])

            with open(os.path.join(topo_feat_dir, csv_file), "r") as f2:
                reader2_list = list(csv.reader(f2))
                header2 = reader2_list[0]
                content2 = reader2_list[1]
                for i in range(len(header2)):
                    if header2[i] not in filtered_header:
                        case_complete_features.append(content2[i])

            merged_csv_content.append(case_complete_features)

        print("the len of merged content is: ", len(merged_csv_content))
        return merged_csv_content

    def write_csv_file(csv_header, csv_content, merged_feat_save_path):
        with open(merged_feat_save_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            for line_content in csv_content:
                writer.writerow(line_content)

    texture_csv_example = os.path.join(texture_feat_dir, list(os.listdir(texture_feat_dir))[0])
    print(texture_csv_example)
    topo_feat_example = os.path.join(topo_feat_dir, list(os.listdir(topo_feat_dir))[0])
    print(topo_feat_example)
    csv_header = merge_csv_header(texture_csv_example, topo_feat_example)
    csv_content = merge_csv_content(texture_feat_dir, topo_feat_dir)
    write_csv_file(csv_header, csv_content, merged_feat_save_path)


if __name__ == '__main__':

    img_base_dir = global_var.base_data_prefix + '/roi'
    roi_mask_base_dir = global_var.base_data_prefix + "/roi_mask"

    generate_texture_features(img_base_dir, roi_mask_base_dir)

    # ------------------------------------------------------------------------------

    topo_mask_base_dir = global_var.base_data_prefix + "/topo_mask"

    print(len(list(os.listdir(topo_mask_base_dir))))
    generate_topology_features(mask_path = topo_mask_base_dir, save_path=global_var.base_sklearn_prefix + "/features_saved/topo_features/")

    # ------------------------------------------------------------------------------
    texture_feat_dir = global_var.base_sklearn_prefix + '/features_saved/texture_features/'
    topo_feat_dir = global_var.base_sklearn_prefix + '/features_saved/topo_features/'
    merged_feat_save_path = global_var.base_sklearn_prefix + '/features_saved/merged_features' + '/merged_features.csv'
    merge_features(texture_feat_dir, topo_feat_dir, merged_feat_save_path)









