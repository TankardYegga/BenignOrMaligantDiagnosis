# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 11/4/2021 6:44 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import csv
import os
import SimpleITK as sitk
from radiomics import featureextractor

from ExtraCode0.generate_mask import find_all_required_files


def generate_single_texture_features(img_path, mask_path, img_label, save_path="./texture_features/"):
    # image2d_arr = cv2.imread(img_path)
    # mask2d_arr = cv2.imread(mask_path)
    print("cur_im:", img_path)
    print("cur mask:", mask_path)
    image2d_arr = sitk.ReadImage(img_path)
    mask2d_arr = sitk.ReadImage(mask_path)
    image2d = sitk.GetImageFromArray(image2d_arr)
    mask2d = sitk.GetImageFromArray(mask2d_arr)
    img_name = os.path.basename(img_path)

    settings = {
        'binWidth': 20,
        'sigma': [1, 2, 3],
        'verbose': True,
        # 'label': 255,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(additionInfo=True, **settings)
    extractor.enableImageTypeByName('LoG')
    extractor.enableImageTypeByName('LBP2D')
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableFeatureClassByName('shape2D')
    result = extractor.execute(image2d, mask2d)
    result["image"] = img_name
    result["label"] = img_label

    popout = list(result.keys())[:10]
    popout.extend(["diagnostics_Mask-original_Hash",
                   "diagnostics_Mask-original_Spacing",
                   "diagnostics_Mask-original_CenterOfMassIndex",
                   "diagnostics_Image-original_Size",
                   "diagnostics_Mask-original_Size",
                   "diagnostics_Mask-original_BoundingBox",
                   "diagnostics_Mask-original_CenterOfMass"
                   ])

    for key in list(result.keys()):
        if key in popout:
            result.pop(key)

    keys, values = [], []
    for key, value in result.items():
        keys.append(key)
        values.append(value)

    with open(save_path + "{}_".format(img_label) + img_name.replace("jpg", "csv"), "w", newline='') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(keys)
        csvwriter.writerow(values)


def generate_texture_features(base_dir, label_info_idx):
    """
    :param base_dir:
    :param label_info_idx: 表示图像标签的关键字在路径中的序号，从0开始计数
    :return:
    """
    file_extensions_list = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG', '.bmp']
    file_path_list = []
    find_all_required_files(file_extensions_list, base_dir, file_path_list)
    file_path_list.sort()
    print(file_path_list)

    for file_idx in range(0, len(file_path_list), 2):
        img_path = file_path_list[file_idx]
        mask_path = file_path_list[file_idx + 1]
        img_label = img_path.split("\\")[label_info_idx]
        generate_single_texture_features(img_path, mask_path, img_label)

base_dir = r'D:\AllExploreDownloads\IDM\Data'
generate_texture_features(base_dir, label_info_idx=6)