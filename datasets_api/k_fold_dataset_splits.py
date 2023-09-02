# -*- encoding: utf-8 -*-
"""
@File    : k_fold_dataset_splits.py
@Time    : 2/16/2022 8:10 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import os
import random
from shutil import copy2

def get_new_series_number(s):
    num_digits = len(str(s))
    zero_prefix = '0' * ( 3 - num_digits)
    s = zero_prefix + str(s)
    return s


# 不能保证训练集和验证集里面的两种分类的标签比例相同
def get_shuffled_data(source_dir='../merged_data/', target_dir='../shuffled_merged_data'):
    # 首先把源文件夹下的所有图片给读取出来，然后获取长度，再对索引进行打乱
    all_image_lists = os.listdir(source_dir)
    num_images = len(all_image_lists)
    img_idx = list(range(num_images))
    random.shuffle(img_idx)
    # print(img_idx)

    # 按照打乱后的索引逐个将图片保存到目标文件夹下面
    for index in range(num_images):
        correspond_shuffled_idx = img_idx[index]
        img_name = all_image_lists[correspond_shuffled_idx]
        new_img_name = get_new_series_number(index) + img_name.split('.')[0] + '.bmp'
        print(new_img_name)
        src_path = os.path.join(source_dir, img_name)
        copy2(src_path, target_dir)
        os.rename(os.path.join(target_dir, img_name), os.path.join(target_dir, new_img_name))


def move_and_rename_file(src_img_name, label, index, source_dir, target_dir):
    target_name = get_new_series_number(index) + src_img_name.split('.')[0] + '.bmp'
    src_path = os.path.join(os.path.join(source_dir, label), src_img_name)
    copy2(src_path, target_dir)
    os.rename(os.path.join(target_dir, src_img_name), os.path.join(target_dir, target_name))


# 尽量保证训练集和验证集里面的两种分类的标签比例相同
def get_shuffled_data_with_label_balanced(source_dir='../data/', target_dir='../shuffled_merged_data2'):
   # 分别获取两类图片数据的长度，然后分别打乱索引
   benign_imgs = os.listdir(os.path.join(source_dir, 'benign'))
   mal_imgs = os.listdir(os.path.join(source_dir, 'malignant'))
   benign_imgs_num = len(benign_imgs)
   mal_imgs_num = len(mal_imgs)
   benign_imgs_idx = list(range(benign_imgs_num))
   random.shuffle(benign_imgs_idx)
   mal_imgs_idx = list(range(mal_imgs_num))
   random.shuffle(mal_imgs_idx)

   # 计算两类数据的长度之比
   if benign_imgs_num <= mal_imgs_num:
        ratio = mal_imgs_num // benign_imgs_num
        flag = 1
   else:
        ratio = benign_imgs_num // mal_imgs_num
        flag = 2

   benign_cursor = 0
   mal_cursor = 0
   total_cursor = 0
   while benign_cursor != benign_imgs_num and mal_cursor != mal_imgs_num:
       if flag == 1:
           cur_benign_img = benign_imgs[benign_cursor]
           move_and_rename_file(cur_benign_img, 'benign', total_cursor, source_dir, target_dir)
           benign_cursor += 1
           total_cursor += 1
           for i in range(ratio):
               cur_mal_img = mal_imgs[mal_cursor]
               move_and_rename_file(cur_mal_img, 'malignant', total_cursor, source_dir, target_dir)
               mal_cursor += 1
               total_cursor += 1
       else:
           cur_mal_img = mal_imgs[mal_cursor]
           move_and_rename_file(cur_mal_img, 'malignant', total_cursor, source_dir, target_dir)
           mal_cursor += 1
           total_cursor += 1
           for i in range(ratio):
               cur_benign_img = benign_imgs[benign_cursor]
               move_and_rename_file(cur_benign_img, 'benign', total_cursor, source_dir, target_dir)
               benign_cursor += 1
               total_cursor += 1

   while benign_cursor != benign_imgs_num:
       cur_benign_img = benign_imgs[benign_cursor]
       move_and_rename_file(cur_benign_img, 'benign', total_cursor, source_dir, target_dir)
       benign_cursor += 1
       total_cursor += 1

   while mal_cursor != mal_imgs_num:
       cur_mal_img = mal_imgs[mal_cursor]
       move_and_rename_file(cur_mal_img, 'malignant', total_cursor, source_dir, target_dir)
       mal_cursor += 1
       total_cursor += 1


# if __name__ == '__main__':
#     # get_shuffled_data()

#     get_shuffled_data_with_label_balanced(source_dir='/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/data_corrected/merged_roi_data',
#                         target_dir='/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/data_corrected/shuffled_merged_data')
