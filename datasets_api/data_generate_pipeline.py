# -*- encoding: utf-8 -*-
"""
@File    : process_test_dataset.py.py
@Time    : 2/26/2022 1:56 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import cv2
import numpy as np
import os
from generate_mask import  generate_mask_by_contour
from generate_masked_roi import get_single_masked_roi
from generate_topo_mask import get_topo_mask
from __init__ import global_var


# 先根据原图生成mask数据
def get_mask_data(img_dir, mask_dir):

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        generate_mask_by_contour(img_path, save_path=mask_dir)
    

# 根据原图和mask生成ROI
def get_roi_data(img_dir, mask_dir, roi_dir, resized_size=128, border_len=5):
     
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name[:-4] + '_output.jpg')
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        save_path = os.path.join(roi_dir, img_name)
        # get_single_masked_roi(img_arr, mask_arr, save_path, True, border_len=5, resized_size=resized_size)
        # get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=False, border_len=5, resized_size=128)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=5, resized_size=128)



# 根据ROI生成Topo Mask
def get_topo_mask_data(roi_dir, topo_mask_dir, threshold=20):
    
    for img_name in os.listdir(roi_dir):
        img_path = os.path.join(roi_dir, img_name)
        topo_mask_path = os.path.join(topo_mask_dir, img_name[:-4] + '_mask.jpg')
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        topo_mask_arr = get_topo_mask(img_arr, threshold)
        cv2.imwrite(topo_mask_path, topo_mask_arr)


# 封装3个步骤为1个流程
def data_generate_pipeline(img_dir, mask_dir, roi_dir, topo_mask_dir, need_topo_mask=True):

    get_mask_data(img_dir, mask_dir)
    get_roi_data(img_dir, mask_dir, roi_dir)
    if need_topo_mask:
        get_topo_mask_data(roi_dir, topo_mask_dir)


def application_case1():
    base_dir = global_var.base_data_prefix2
    img_dir = base_dir + '/whole'
    mask_dir = base_dir + '/mask'
    roi_dir = base_dir + '/roi'
    topo_mask_dir = base_dir + '/topo_mask'

    data_generate_pipeline(img_dir, mask_dir, roi_dir, topo_mask_dir)

    # get_roi_data(img_dir, mask_dir, roi_dir)
    # get_topo_mask_data(roi_dir=roi_dir, topo_mask_dir=topo_mask_dir)


    # roi_128_dir = base_dir + '/roi_128'
    # topo_mask_128_dir = base_dir + '/topo_mask_128'
    # get_topo_mask_data(roi_dir=roi_128_dir, topo_mask_dir=topo_mask_128_dir)

    # roi_dir = base_dir + '/roi'
    # topo_mask_th10_dir = base_dir + '/topo_mask_th10'
    # get_topo_mask_data(roi_dir=roi_dir, topo_mask_dir=topo_mask_th10_dir, threshold=10)

    # roi_dir = base_dir + '/roi'
    # topo_mask_th15_dir = base_dir + '/topo_mask_th15'
    # get_topo_mask_data(roi_dir=roi_dir, topo_mask_dir=topo_mask_th15_dir, threshold=15)
    
    
# 在测试数据上生成roi和topo_mask
def application_case2_on_test_data():

    img_dir = global_var.base_test_data_prefix + '/whole'
    mask_dir = global_var.base_test_data_prefix + '/mask'
    roi_dir = global_var.base_test_data_prefix + '/roi'
    topo_mask_dir = global_var.base_test_data_prefix + '/topo_mask'

    # get_roi_data(img_dir, mask_dir, roi_dir, resized_size=256, border_len=8)
    get_topo_mask_data(roi_dir, topo_mask_dir, threshold=20)




# 在指定数据上生成roi和topo_mask
def application_case3():

    img_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/whole'
    mask_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/mask'
    roi_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/original_roi'
    topo_mask_dir = ''

    data_generate_pipeline(img_dir, mask_dir, roi_dir, topo_mask_dir, need_topo_mask=False)


# 在指定数据上生成roi和topo_mask
def application_case4():

    img_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/whole'
    mask_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/mask'
    roi_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/roi'
    topo_mask_dir = ''

    data_generate_pipeline(img_dir, mask_dir, roi_dir, topo_mask_dir, need_topo_mask=False)


if __name__ == '__main__':

    # application_case1();

    # application_case2_on_test_data();

    # application_case3()

    application_case4()
    












    

