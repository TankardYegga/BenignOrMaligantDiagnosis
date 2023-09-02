# -*- encoding: utf-8 -*-
"""
@File    : dataset_split.py
@Time    : 11/30/2021 2:26 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
# 工具类
import os
import random
import shutil
from shutil import copy2,copy


# 划分的train、val、test的下面需要再划分benign和malignant两个子文件夹
def dataset_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.2, test_scale=0.0):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)


    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        # 首先获取该类下面的所有图片
        # 使用图片数组来存储，该数组中图片的顺序与图片在文件资源管理器中的顺序是一致的
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        # 打乱数组的下标，其实也就是得到一个随机访问数组元素的顺序
        random.shuffle(current_data_index_list)

        # 对每个类的图片进行训练、验证和测试的划分
        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = int(current_data_length * train_scale)
        val_stop_flag = int(current_data_length * (train_scale + val_scale))
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0

        # 按照这个随机顺序访问图片
        # 按照访问的顺序前X个作为训练集，中间Y个作为验证集，结尾Z个作为测试集
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx < train_stop_flag:
                copy2(src_img_path, train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            elif (current_idx >= train_stop_flag) and (current_idx < val_stop_flag):
                copy2(src_img_path, val_folder)
                # print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1
            else:
                copy2(src_img_path, test_folder)
                # print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))



# 划分的train、val、test的下面不再划分benign和malignant两个子文件夹
def dataset_split2(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.2, test_scale=0.0):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        split_path = os.path.join(split_path, 'roi/')
        if os.path.isdir(split_path):
            pass
        else:
            os.makedirs(split_path)


    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        # 首先获取该类下面的所有图片
        # 使用图片数组来存储，该数组中图片的顺序与图片在文件资源管理器中的顺序是一致的
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        # 打乱数组的下标，其实也就是得到一个随机访问数组元素的顺序
        random.shuffle(current_data_index_list)

        # 对每个类的图片进行训练、验证和测试的划分
        train_folder = os.path.join(os.path.join(target_data_folder, 'train'),'roi')
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), 'roi')
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), 'roi')

        train_stop_flag = int(current_data_length * train_scale)
        val_stop_flag = int(current_data_length * (train_scale + val_scale))
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0

        # 按照这个随机顺序访问图片
        # 按照访问的顺序前X个作为训练集，中间Y个作为验证集，结尾Z个作为测试集
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx < train_stop_flag:
                copy(src_img_path, train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            elif (current_idx >= train_stop_flag) and (current_idx < val_stop_flag):
                copy(src_img_path, val_folder)
                # print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1
            else:
                copy(src_img_path, test_folder)
                # print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1


        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))



def get_filtered_data(data_path, filtered_name='mask'):
    imgs = os.listdir(data_path)
    filtered_imgs = []
    for img in imgs:
        if filtered_name in img:
            continue
        filtered_imgs.append(img)
    return filtered_imgs



# 划分的train、val、test的下面不再划分benign、normal、malignant三个文件夹, 且train或者test或者val下面会分为原图和标签两个子文件夹下
def dataset_split3(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.2, test_scale=0.0):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)

    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        img_split_path = os.path.join(split_path, 'img/')
        
        mask_split_path = os.path.join(split_path, 'mask/')

        if os.path.isdir(img_split_path):
            pass
        else:
            os.makedirs(img_split_path)

        if os.path.isdir(mask_split_path):
            pass
        else:
            os.makedirs(mask_split_path)


    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        # 首先获取该类下面的所有图片
        # 使用图片数组来存储，该数组中图片的顺序与图片在文件资源管理器中的顺序是一致的
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = get_filtered_data(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        # 打乱数组的下标，其实也就是得到一个随机访问数组元素的顺序
        random.shuffle(current_data_index_list)

        # 对每个类的图片进行训练、验证和测试的划分
        img_train_folder = os.path.join(os.path.join(target_data_folder, 'train'),'img')
        mask_train_folder = os.path.join(os.path.join(target_data_folder, 'train'),'mask')

        img_val_folder = os.path.join(os.path.join(target_data_folder, 'val'), 'img')
        mask_val_folder = os.path.join(os.path.join(target_data_folder, 'val'),'mask')

        img_test_folder = os.path.join(os.path.join(target_data_folder, 'test'), 'img')
        mask_test_folder = os.path.join(os.path.join(target_data_folder, 'test'),'mask')

        train_stop_flag = int(current_data_length * train_scale)
        val_stop_flag = int(current_data_length * (train_scale + val_scale))
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0

        # 按照这个随机顺序访问图片
        # 按照访问的顺序前X个作为训练集，中间Y个作为验证集，结尾Z个作为测试集
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            cur_img_name = current_all_data[i].split('.')[0]
            src_mask_path = os.path.join(current_class_data_path, cur_img_name + '_mask.png')

            if current_idx < train_stop_flag:
                copy(src_img_path, img_train_folder)
                copy(src_mask_path, mask_train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            elif (current_idx >= train_stop_flag) and (current_idx < val_stop_flag):
                copy(src_img_path, img_val_folder)
                copy(src_mask_path, mask_val_folder)
                # print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1
            else:
                copy(src_img_path, img_test_folder)
                copy(src_mask_path, mask_test_folder)
                # print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1


        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(img_train_folder, train_num))
        print("验证集{}：{}张".format(img_val_folder, val_num))
        print("测试集{}：{}张".format(img_test_folder, test_num))




if __name__ == '__main__':
    #   src_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/original_data/classified_roi'
    #   target_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data'
    #   dataset_split(src_data_folder, target_data_folder, 0.8, 0.2)

    #   src_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected/classified_roi'
    #   target_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data2'
    #   dataset_split(src_data_folder, target_data_folder, 0.9, 0.1)

    
    #   src_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/classified_roi/'
    #   target_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_1'
    #   dataset_split2(src_data_folder, target_data_folder, 0.8, 0.1, 0.1)


    #   src_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/classified_roi/'
    #   target_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_2'
    #   dataset_split2(src_data_folder, target_data_folder, 0.8, 0.1, 0.1)

    #   src_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/classified_roi/'
    #   target_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_3/'
    #   dataset_split2(src_data_folder, target_data_folder, 0.8, 0.1, 0.1)

    #   src_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/classified_roi/'
    #   target_data_folder = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4'
    #   dataset_split2(src_data_folder, target_data_folder, 0.9, 0.1, 0.0)

      src_data_folder = '/mnt/520_v2/lxy/Dataset_BUSI_with_GT/'
      target_data_folder = '/mnt/520_v2/lxy/Breast_US_Seg/'
      dataset_split3(src_data_folder, target_data_folder, 0.9, 0.0, 0.1)


#     src_data_folder = r"D:\AllExploreDownloads\IDM\ExtraCode\data"
#     target_data_folder = r"D:\AllExploreDownloads\IDM\ExtraCode\split_data"
#     dataset_split(src_data_folder, target_data_folder)

#     src_data_folder = r"/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/data_corrected/classified_roi"
#     target_data_folder = r"/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/data_corrected/fixed_split_data"
#     dataset_split(src_data_folder, target_data_folder)

#     src_data_folder = r"/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/data_corrected/classified_topo_mask"
#     target_data_folder = r"/mnt/520/lijingyu/lijingyu/zlw/ExtraCode/data_corrected/fixed_split_topo_mask"
#     dataset_split(src_data_folder, target_data_folder)


