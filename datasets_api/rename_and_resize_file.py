# -*- encoding: utf-8 -*-
"""
@File    : rename_file.py.py
@Time    : 11/3/2021 9:14 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import copy
import os
import re
import sys

import cv2
import shutil
from generate_mask import find_all_required_files
import numpy as np
from generate_masked_roi import get_single_masked_roi
from __init__ import global_var

def rename_file(base_dir, replaced_words, replacing_list, enable_dir = True):
    """
    :param base_dir:
    :param replaced_words: 需要被替换的中文词汇
    :param replacing_list:  替换的英文词汇
    :param enable_dir: 判断是否需要对文件夹重命名
    :return:
    """
    replaced_words_dict = dict(zip(replaced_words, replacing_list))
    for file in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, file)):
            # 判断当前目录名称是否是可替换词语数组中某个词语的子串
            for replaced_word in replaced_words:
                if replaced_word in file and enable_dir:
                    new_file = replaced_words_dict[replaced_word]
                    print("file: ", file)
                    print("new file: ", new_file)
                    os.rename(os.path.join(base_dir, file), os.path.join(base_dir, new_file))
                    file = new_file
                    break
            rename_file(base_dir + '\\' + file, replaced_words, replacing_list, enable_dir)

        elif os.path.isfile(os.path.join(base_dir, file)):
            for replaced_word in replaced_words:
                if replaced_word in file:
                    file_suffix = os.path.splitext(file)[-1]
                    new_file = replaced_words_dict[replaced_word] + file_suffix
                    os.rename(os.path.join(base_dir, file), os.path.join(base_dir, new_file))
                    break
        else:
            print("error occurs")


def index_number(path=''):
    kv = []
    nums = []
    beforeDatas = re.findall('\d', path)
    for num in beforeDatas:
        indexV = []
        times = path.count(num)
        if (times > 1):
            if (num not in nums):
                indexs = re.finditer(num, path)
                for index in indexs:
                    iV = []
                    i = index.span()[0]
                    iV.append(num)
                    iV.append(i)
                    kv.append(iV)
            nums.append(num)
        else:
            index = path.find(num)
            indexV.append(num)
            indexV.append(index)
            kv.append(indexV)
    # 根据数字位置排序
    indexSort = []
    resultIndex = []
    for vi in kv:
        indexSort.append(vi[1])
    indexSort.sort()
    for i in indexSort:
        for v in kv:
            if (i == v[1]):
                resultIndex.append(v)
    return resultIndex


def test_index_number():
    path = 'B1RCC'
    print(index_number(path))

    path2 = 'a2aa2bbb3ccc4dddd'
    print(index_number(path2))


def find_number(file, separator='_'):
    num_digit = 1
    res = re.findall(separator + '\d'* num_digit, file)
    while len(res) != 0:
        num_digit += 1
        last_res = res
        res = re.findall(separator + '\d' * num_digit, file)

    num_indexer = list(re.finditer(last_res[0], file))[0]
    num_start_idx = num_indexer.span()[0]
    num_end_idx = num_indexer.span()[1]

    return last_res[0], num_start_idx, num_end_idx


def is_matched_file(file1, file2):
    file1_name = os.path.basename(file1)
    file2_name = os.path.basename(file2)
    label1 = file1_name[0]
    label2 = file2_name[0]
    case_num1, num_start_idx1, num_end_idx1 = find_number(file1_name)
    case_num2, num_start_idx2, num_end_idx2 = find_number(file2_name, '')
    img_type1 = file1_name[num_end_idx1:]
    img_type2 = file2_name[num_end_idx2:]

    common_name = ''
    is_matched = label1 == label2 and case_num1[1:] == case_num2 and \
                 ( img_type1 in img_type2 or img_type2 in img_type1 )
    if is_matched:
        common_name = file1_name[:num_start_idx1] + case_num1 + \
                      img_type2 if img_type1 in img_type2 else img_type1
    return is_matched, common_name


def set_consistent_name(base_dir1, base_dir2):
    """
    对两个文件夹下的文件进行统一命名
    :param base_dir1:
    :param base_dir2:
    :return:
    """
    # 首先找到两个文件夹下的所有文件
    file_extensions_list = ['.csv']
    file_path_list_1 = []
    file_path_list_2 = []
    find_all_required_files(file_extensions_list, base_dir1, file_path_list_1)
    find_all_required_files(file_extensions_list, base_dir2, file_path_list_2)

    # 计算目录2中与目录1中指定文件相匹配的文件路径
    # 计算两个文件名的最终统一命名
    # 对两个目录下的对应文件进行重命名
    match_finished = [False] * len(file_path_list_2)
    match_finished_dict = dict(zip(file_path_list_2, match_finished))
    for file1 in file_path_list_1:
        for file2 in file_path_list_2:
            if not match_finished_dict[file2]:
                is_matched, common_name = is_matched_file(file1, file2)
                if is_matched:
                    match_finished_dict[file2] = True
                    common_name_path1 = os.path.join(os.path.dirname(file1), common_name)
                    common_name_path2 = os.path.join(os.path.dirname(file2), common_name)
                    # if not os.path.exists(common_name_path1):

                    if file1 != common_name_path1:
                        os.rename(file1, common_name_path1)
                    if file2 != common_name_path2:
                        os.rename(file2, common_name_path2)
                    break


def resize_topo_mask_images(mask_img_dir, resized_mask_dir, resized_size=256):
    for img in os.listdir(mask_img_dir):
        img_path = os.path.join(mask_img_dir, img)
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        resized_img_arr = cv2.resize(img_arr, (resized_size, resized_size), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(resized_mask_dir, img), resized_img_arr)


def find_data_with_no_suffix(cur_path, file_path_list):
    # 千万不要在循环里面做可能影响循环值的问题
    for file_or_dir in os.listdir(cur_path):
        cur_sub_path = os.path.join(cur_path, file_or_dir)
        print('cur path:', cur_sub_path)
        if os.path.isfile(cur_sub_path):
            cur_file_ext = os.path.splitext(os.path.basename(cur_sub_path))
            # print('cur file ext', cur_file_ext)
            # 如果文件没有后缀名，不再处理
            if cur_file_ext[1] == '':
                file_path_list.append(cur_sub_path)
                # print(cur_path)
        elif os.path.isdir(cur_sub_path):
            # print('go deeper:')
            find_data_with_no_suffix(cur_sub_path, file_path_list)
        else:
            continue


def obtain_dicom_data(source_path = r'D:\AllExploreDownloads\IDM\Data',
                      save_dir = r'D:\AllExploreDownloads\IDM\DicomData'):

    file_path_list = []
    find_data_with_no_suffix(source_path, file_path_list)
    print(len(file_path_list))
    print(file_path_list)

    for dicom_file in file_path_list:
        dicom_file_name = os.path.basename(dicom_file)
        if 'DICOMDIR' in dicom_file_name:
            continue
        # print("0", dicom_file)
        # print("1", dicom_file_name)
        with open(dicom_file, 'rb') as fstream:
            content = fstream.read()
            new_dicom_save_path = os.path.join(save_dir, dicom_file_name + '.dcm')
            with open(new_dicom_save_path, 'wb') as wstream:
                wstream.write(content)


if __name__ == '__main__':
    # replaced_words = ['恶性病变', '良性病变']
    # replacing_words = ['Malignant', 'Benign']
    # base_dir = r'D:\AllExploreDownloads\IDM\Data'
    # rename_file(base_dir, replaced_words, replacing_words, enable_dir = True)

    # base_dir1 = r'D:\AllExploreDownloads\IDM\ExtraCode0\texture_features'
    # base_dir2 = r'D:\AllExploreDownloads\IDM\ExtraCode0\topo_features'
    # set_consistent_name(base_dir1, base_dir2)
    # test1()

    # source_path = 'D:\\AllExploreDownloads\\IDM\\Data'
    # file_extensions_list = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG', '.bmp']
    # file_path_list = []
    # find_all_required_files(file_extensions_list, source_path, file_path_list)
    # print(file_path_list)
    #
    # is_resized = True
    # if is_resized:
    #     save_dir = r'D:\AllExploreDownloads\IDM\ResizedImages'
    # else:
    #     save_dir = r'D:\AllExploreDownloads\IDM\OriginalImages'
    # if len(os.listdir(save_dir)):
    #     shutil.rmtree(save_dir)
    #     os.mkdir(save_dir)
    #
    # count = 0
    # for i in range(0, len(file_path_list), 2):
    #     img_path = file_path_list[i]
    #     mask_path = file_path_list[i+1]
    #
    #     count += 1
    #
    #     print(img_path)
    #     img_name = os.path.basename(img_path).split('.')[0]
    #     print(img_name)
    #     if 'Benign' in img_path:
    #         img_name = 'B' + img_name + '.bmp'
    #     else:
    #         img_name = 'M' + img_name + '.bmp'
    #     print(img_name)
    #     save_path = os.path.join(save_dir, img_name)
    #     print(save_path)
    #
    #     img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #     mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    #     get_masked_roi(img_arr, mask_arr, save_path, is_resized)
    #
    # print('count is', count)

    # obtain_dicom_data()

    # mask_img_dir =  global_var.base_data_prefix + '/roi_128'
    # resized_mask_dir = global_var.base_data_prefix + '/roi'
    # resize_topo_mask_images(mask_img_dir, resized_mask_dir)
    
    # mask_img_dir =  global_var.base_data_api_prefix + '/test_data' 
    # resized_mask_dir =  global_var.base_data_api_prefix + '/test_data2' 
    # resize_topo_mask_images(mask_img_dir, resized_mask_dir)

    # mask_img_dir =  global_var.base_data_prefix + '/topo_mask_128'
    # resized_mask_dir = global_var.base_data_prefix + '/topo_mask_256'
    # resize_topo_mask_images(mask_img_dir, resized_mask_dir)

    # mask_img_dir =  global_var.base_test_data_prefix + '/roi'
    # resized_mask_dir = global_var.base_test_data_prefix + '/roi_224'
    # resize_topo_mask_images(mask_img_dir, resized_mask_dir)

    # mask_img_dir =  global_var.base_ljy_prefix + '/tests'
    # resized_mask_dir = global_var.base_ljy_prefix + '/tests_256'
    # resize_topo_mask_images(mask_img_dir, resized_mask_dir)

    # mask_img_dir =  global_var.base_ljy_prefix + '/trains_224'
    # resized_mask_dir = global_var.base_ljy_prefix + '/trains_256'
    # resize_topo_mask_images(mask_img_dir, resized_mask_dir)

    # mask_img_dir =  global_var.base_data_trains256_prefix + '/roi/'
    # resized_mask_dir = global_var.base_data_trains256_prefix + '/roi/'
    # resize_topo_mask_images(mask_img_dir, resized_mask_dir)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi'
    # resized_dir =   '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/resized_train_part_of_original_data/roi'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=224)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/topo_mask2'
    # resized_dir =  '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/topo_mask2_16'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=16)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/ori_size_roi/'
    # resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/roi_64/'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=64)


    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/ori_size_roi/'
    # resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/roi_128/'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=128)


    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test2/original_roi'
    # resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test2/roi'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=128)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/train/roi/'
    # resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/train/roi_256/'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=256)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/roi/'
    # resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/roi_256/'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=256)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test2/roi/'
    # resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test2/roi_256/'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=256)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/roi/'
    # resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/roi_256/'
    # resize_topo_mask_images(img_dir, resized_dir, resized_size=256)


    img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/train/roi/'
    resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/train/roi_299/'
    resize_topo_mask_images(img_dir, resized_dir, resized_size=299)

    img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/test/roi/'
    resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/test/roi_299/'
    resize_topo_mask_images(img_dir, resized_dir, resized_size=299)

    img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/val/roi/'
    resized_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/val/roi_299/'
    resize_topo_mask_images(img_dir, resized_dir, resized_size=299)










