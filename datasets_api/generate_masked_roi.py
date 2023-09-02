from tkinter import N, mainloop
import cv2
from pip import main
from __init__ import global_var
import os
import numpy as np


# 单张图片 + Mask = 单张图片的ROI
def get_single_masked_roi(original_img_arr, original_mask_arr, save_path, is_resized=True, border_len=5, resized_size=128):

    # img_arr = copy.deepcopy(original_img_arr)
    # mask_arr = copy.deepcopy(original_mask_arr)
    img_arr = original_img_arr
    mask_arr = original_mask_arr
    print(img_arr.shape)
    print(img_arr.dtype)

    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_RGB2GRAY)
    print(img_arr.shape)
    print(img_arr.dtype)

    min_row_idx = mask_arr.shape[0]
    max_row_idx = -1
    min_col_idx = mask_arr.shape[-1]
    max_col_idx = -1
    count = 0
    for i in range(mask_arr.shape[0]):
        for j in range(mask_arr.shape[-1]):
            if mask_arr[i][j] == 255:
            # if mask_arr[i][j] == 255 or mask_arr[i][j] == 244 :
            # if mask_arr[i][j] > 240:
                count += 1
                # print(i, ":", j)
                if i < min_row_idx:
                    min_row_idx = i
                if i > max_row_idx:
                    max_row_idx = i
                if j < min_col_idx:
                    min_col_idx = j
                if j > max_col_idx:
                    max_col_idx = j

    print(max_row_idx)
    print(min_row_idx)

    print(max_col_idx)
    print(min_col_idx)

    min_row_idx += border_len
    max_row_idx -= border_len
    min_col_idx += border_len
    max_col_idx -= border_len

    row_len = max_row_idx - min_row_idx + 1
    col_len = max_col_idx - min_col_idx + 1
    print(count)
    print(row_len)
    print(col_len)
    print(row_len * col_len == count)

    # 如果不加数据类型，这里默认是float64，而float64在使用cvtColor的时候会报错
    masked_img_arr = np.zeros((row_len, col_len), dtype=np.uint8)
    for i in range(min_row_idx, max_row_idx+1):
        for j in range(min_col_idx, max_col_idx+1):
            masked_img_arr[i-min_row_idx][j-min_col_idx] = img_arr[i][j]

    masked_img_arr = cv2.cvtColor(masked_img_arr, cv2.COLOR_GRAY2BGR)
    print('masked img shape:', masked_img_arr.shape)
    if is_resized:
        masked_img_arr = cv2.resize(masked_img_arr, (resized_size, resized_size), interpolation=cv2.INTER_CUBIC)
        print('resized masked img shape:', masked_img_arr.shape)

    cv2.imwrite(save_path, masked_img_arr)
    print('saved!')


# 一些图片提取ROI后仍然有边框，需要设置不同的border_len来解决
def test_single_img_roi():
    img_list = ['M99RMLO.jpg',  'M100RMLO.jpg', 'B17LMLO.jpg',]
    for img in img_list:
        img_name = img[:-4]
        img_path = global_var.base_data_prefix + '/whole/' + img
        mask_path = global_var.base_data_prefix + '/mask/' + img_name + '_output.jpg'

        save_path = global_var.base_data_api_prefix + '/test_data/' + img_name + '_mask.jpg'
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=9)
    
    img_list = ['M99RCC.jpg',  'M100RCC.jpg',]
    for img in img_list:
        img_name = img[:-4]
        img_path = global_var.base_data_prefix + '/whole/' + img
        mask_path = global_var.base_data_prefix + '/mask/' + img_name + '_output.jpg'

        save_path = global_var.base_data_api_prefix + '/test_data/' + img_name + '_mask.jpg'
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=15)


    img_list = ['B4RMLO.jpg']
    for img in img_list:
        img_name = img[:-4]
        img_path = global_var.base_data_prefix + '/whole/' + img
        mask_path = global_var.base_data_prefix + '/mask/' + img_name + '_output.jpg'

        save_path = global_var.base_data_api_prefix + '/test_data/' + img_name + '_mask.jpg'
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=17)

    img_list = ['M73RMLO.jpg', 'M73RCC.jpg',  'M72LCC.jpg']
    for img in img_list:
        img_name = img[:-4]
        img_path = global_var.base_data_prefix + '/whole/' + img
        mask_path = global_var.base_data_prefix + '/mask/' + img_name + '_output.jpg'

        save_path = global_var.base_data_api_prefix + '/test_data/' + img
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=12)

    img_list = ['M66LMLO.jpg', ]
    for img in img_list:
        img_name = img[:-4]
        img_path = global_var.base_data_prefix + '/whole/' + img
        mask_path = global_var.base_data_prefix + '/mask/' + img_name + '_output.jpg'

        save_path = global_var.base_data_api_prefix + '/test_data/' + img
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=14)

    img_list = ['M72LMLO.jpg', ]
    for img in img_list:
        img_name = img[:-4]
        img_path = global_var.base_data_prefix + '/whole/' + img
        mask_path = global_var.base_data_prefix + '/mask/' + img_name + '_output.jpg'

        save_path = global_var.base_data_api_prefix + '/test_data/' + img
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=18)

    img_list = ['M98LMLO.jpg', ]
    for img in img_list:
        img_name = img[:-4]
        img_path = global_var.base_data_prefix + '/whole/' + img
        mask_path = global_var.base_data_prefix + '/mask/' + img_name + '_output.jpg'

        save_path = global_var.base_data_api_prefix + '/test_data/' + img
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=10)


    img_list = [ 'M23LMLO.jpg']
    for img in img_list:
        img_name = img[:-4]
        img_path = global_var.base_data_prefix + '/whole/' + img
        mask_path = global_var.base_data_prefix + '/mask/' + img_name + '_output.jpg'

        save_path = global_var.base_data_api_prefix + '/test_data/' + img
        img_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        get_single_masked_roi(img_arr, mask_arr, save_path, is_resized=True, border_len=15)

    
if __name__ == '__main__':
    test_single_img_roi()
