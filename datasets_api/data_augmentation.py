# -*- coding:utf-8 -*-

"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
"""
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
import math
import shutil
import sys
from cv2.cv2 import imwrite
from matplotlib import pyplot as plt

from __init__ import global_var

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True



class DataAugmentation:
    """
    包含数据增强的八种方式
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r") # rgb模式

    @staticmethod
    def randomFlip(image):
        """
        对图像进行上下左右四个方面的随机翻转
        :param image: PIL的图像image
        :param model: 水平或者垂直方向的随机翻转模式,默认右向翻转
        :return: 翻转之后的图像
        """
        random_model = np.random.randint(0, 2)
        flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        flipimage=image
        flipimage = flipimage.transpose(flip_model[random_model])
        flipimage = cv2.cvtColor(np.asarray(flipimage), cv2.COLOR_RGB2BGR)
        flipimage = cv2.resize(flipimage, (256, 256))
        return flipimage
        # return image.transpose(mode)

    @staticmethod
    def randomShift(image):
        # def randomShift(image, xoffset, yoffset=None):
        """
        对图像进行平移操作
        :param image: PIL的图像image
        :param xoffset: x方向向右平移
        :param yoffset: y方向向下平移
        :return: 翻转之后的图像
        """
        random_xoffset = np.random.randint(0, math.ceil(image.size[0] * 0.2))
        random_yoffset = np.random.randint(0, math.ceil(image.size[1] * 0.2))
        # return image.offset(xoffset = random_xoffset, yoffset = random_yoffset)
        return image.offset(random_xoffset)

    @staticmethod
    def rotate_image(image):
        """
        angle: 旋转的角度
        crop: 是否需要进行裁剪，布尔向量
        """

        angle = np.random.randint(10, 90)
        crop=1
        # 去除黑边的操作
        crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w]  # 定义裁切函数，后续裁切黑边使用
        w, h = image.size[:2]
        # 旋转角度的周期是360°
        angle %= 360
        # 计算仿射变换矩阵
        M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        # 得到旋转后的图像
        img_rgb2 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        img_rotated = cv2.warpAffine(img_rgb2, M_rotation, (w, h))

        # 如果需要去除黑边
        if crop:
            # 裁剪角度的等效周期是180°
            angle_crop = angle % 180
            if angle > 90:
                angle_crop = 180 - angle_crop
            # 转化角度为弧度
            theta = angle_crop * np.pi / 180
            # 计算高宽比
            hw_ratio = float(h) / float(w)
            # 计算裁剪边长系数的分子项
            tan_theta = np.tan(theta)
            numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

            # 计算分母中和高宽比相关的项
            r = hw_ratio if h > w else 1 / hw_ratio
            # 计算分母项
            denominator = r * tan_theta + 1
            # 最终的边长系数
            crop_mult = numerator / denominator

            # 得到裁剪区域
            w_crop = int(crop_mult * w)
            h_crop = int(crop_mult * h)
            x0 = int((w - w_crop) / 2)
            y0 = int((h - h_crop) / 2)
            img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
            # img_rotated = Image.fromarray(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
            # img=ImageTK.PhotoImage(img_rotated2)
            img_rotated = cv2.resize(img_rotated, (256, 256))
            # img_rotated = img_rotated[:, :, ::-1]
            # 分离出B,G,R三个通道
            # b, g, r = cv2.split(img_rotated)
            # # # 交换B和R的位置再组合
            # img_rotated = cv2.merge((r, g, b))
            # # # 此时B和R通道的数值交换了，但是通道标记还是BGR
            # # # 变更通道模式
            # img_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
            # 此时再输出就是正确的RBG图像了
            # img_rotated = img_rotated[..., ::-1]
            # plt.figure(figsize=(15, 10))
            # plt.subplot(2 ,2, 1), plt.imshow(image)
            # plt.axis('off');
            # plt.subplot(2,2, 2), plt.imshow(img_rotated)
            # plt.axis('off');
            # plt.show()
        return img_rotated

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """

        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,裁剪图像大小宽和高的5/6
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        img=image
        image_width = image.size[0]
        image_height = image.size[1]
        crop_image_width = math.ceil(image_width * 7 / 10)
        crop_image_height = math.ceil(image_height * 7 / 10)
        x = np.random.randint(0, image_width - crop_image_width)
        y = np.random.randint(0, image_height - crop_image_height)
        random_region = (x, y, x + crop_image_width, y + crop_image_height)
        image_crop=image.crop(random_region)
        image_crop = cv2.cvtColor(np.asarray(image_crop), cv2.COLOR_RGB2BGR)
        image_crop = cv2.resize(image_crop, (256, 256))
        # DataAugmentation.saveImage(image, filename)
        # plt.figure(figsize=(15, 10))
        # plt.subplot(2 ,2, 1), plt.imshow(image)
        # plt.axis('off');
        # plt.subplot(2, 2,2), plt.imshow(image_crop)
        # plt.axis('off');
        # plt.show()
        return image_crop

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        try:
            img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
            img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
            img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
        except:
            img = img
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def saveImage(image, path):
        try:
            image.save(path)
        except Exception as e:
            print(e)
            print('not save img: ', path)
            pass

files = []

def get_files(dir_path):
    global files
    if os.path.exists(dir_path):
        parents = os.listdir(dir_path)
        for parent in parents:
            child = os.path.join(dir_path, parent)
            if os.path.exists(child) and os.path.isfile(child):
                # child = child.split('/')[4:]
                # str_child = '/'.join(child)
                files.append(child)
            elif os.path.isdir(child):
                get_files(child)
        return files
    else:
        return None


def copy_data_between_dirs(imgs_dir, imgs_moved_to_dir):

    for img in os.listdir(imgs_dir):
        img_path = os.path.join(imgs_dir, img)
        shutil.copy2(img_path, imgs_moved_to_dir)
    pass



def aug_app1(times):

    imgs_dir = global_var.base_data_prefix + '/roi'
    imgs_aug_dir = global_var.base_data_aug_prefix + '/roi'

    copy_data_between_dirs(imgs_dir, imgs_aug_dir)

    saved_idx_dict = {'B': 120, 'M': 120}
   
    funcMap = {"flip": DataAugmentation.randomFlip,
               "rotation": DataAugmentation.rotate_image,
               "crop": DataAugmentation.randomCrop,
               "color": DataAugmentation.randomColor,
               "gaussian": DataAugmentation.randomGaussian,
               "shift": DataAugmentation.randomShift
               }

    # funcLists = {"flip", "rotation", "crop", "color", "gaussian"}
    funcLists = ["flip", "rotation", "crop"]

    imgs_list = get_files(imgs_dir)

    times = times
    
    for time in range(times):

        for index_img, img_path in enumerate(imgs_list):

            postfix = img_path.split('.')[1]  # 后缀
            
            img_name = os.path.basename(img_path)

            saved_idx_dict[img_name[0]] += 1

            saved_img_name = img_name[0] + str(saved_idx_dict[img_name[0]]) + '.' + postfix
            saved_img_path = os.path.join(imgs_aug_dir, saved_img_name)

            if postfix.lower() in ['jpg', 'jpeg', 'png', 'bmp']:

                image = DataAugmentation.openImage(img_path)

                random_func = np.random.randint(0, 3)

                flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
                func=funcLists[random_func]

                if func == 'flip':
                    new_image_arr = DataAugmentation.randomFlip(image)
                    
                if func == 'rotation':
                    new_image_arr = DataAugmentation.rotate_image(image)

                elif func == 'crop':
                    new_image_arr = DataAugmentation.randomCrop(image)

                else:
                    new_image_arr = funcMap[func](image)
                
                new_image = Image.fromarray(new_image_arr, mode='RGB')
                DataAugmentation.saveImage(new_image, saved_img_path)



def aug_app2(times):

    imgs_dir = global_var.base_data_prefix + '/roi'
    imgs_aug_dir = global_var.base_data_aug_2_prefix + '/roi'

    copy_data_between_dirs(imgs_dir, imgs_aug_dir)

    saved_idx_dict = {'B': 120, 'M': 120}
   
    funcMap = {"flip": DataAugmentation.randomFlip,
               "rotation": DataAugmentation.rotate_image,
               "crop": DataAugmentation.randomCrop,
               "color": DataAugmentation.randomColor,
               "gaussian": DataAugmentation.randomGaussian,
               "shift": DataAugmentation.randomShift
               }

    # funcLists = {"flip", "rotation", "crop", "color", "gaussian"}
    funcLists = ["flip", "rotation", "crop"]

    imgs_list = get_files(imgs_dir)

    times = times
    
    for time in range(times):

        for index_img, img_path in enumerate(imgs_list):

            postfix = img_path.split('.')[1]  # 后缀
            
            img_name = os.path.basename(img_path)

            saved_idx_dict[img_name[0]] += 1

            saved_img_name = img_name[0] + str(saved_idx_dict[img_name[0]]) + '.' + postfix
            saved_img_path = os.path.join(imgs_aug_dir, saved_img_name)

            if postfix.lower() in ['jpg', 'jpeg', 'png', 'bmp']:

                image = DataAugmentation.openImage(img_path)

                random_func = np.random.randint(0, 3)

                flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
                func=funcLists[random_func]

                if func == 'flip':
                    new_image_arr = DataAugmentation.randomFlip(image)
                    
                if func == 'rotation':
                    new_image_arr = DataAugmentation.rotate_image(image)

                elif func == 'crop':
                    new_image_arr = DataAugmentation.randomCrop(image)

                else:
                    new_image_arr = funcMap[func](image)
                
                new_image = Image.fromarray(new_image_arr, mode='RGB')
                DataAugmentation.saveImage(new_image, saved_img_path)



def aug_app3(times):

    imgs_dir = global_var.base_data_prefix + '/roi'
    imgs_aug_dir = global_var.base_data_aug_3_prefix + '/roi'

    copy_data_between_dirs(imgs_dir, imgs_aug_dir)

    saved_idx_dict = {'B': 120, 'M': 120}
   
    funcMap = {"flip": DataAugmentation.randomFlip,
               "rotation": DataAugmentation.rotate_image,
               "crop": DataAugmentation.randomCrop,
               "color": DataAugmentation.randomColor,
               "gaussian": DataAugmentation.randomGaussian,
               "shift": DataAugmentation.randomShift
               }

    # funcLists = {"flip", "rotation", "crop", "color", "gaussian"}
    funcLists = ["flip", "rotation", "crop"]

    imgs_list = get_files(imgs_dir)

    times = times
    
    for time in range(times):

        for index_img, img_path in enumerate(imgs_list):

            postfix = img_path.split('.')[1]  # 后缀
            
            img_name = os.path.basename(img_path)

            saved_idx_dict[img_name[0]] += 1

            saved_img_name = img_name[0] + str(saved_idx_dict[img_name[0]]) + '.' + postfix
            saved_img_path = os.path.join(imgs_aug_dir, saved_img_name)

            if postfix.lower() in ['jpg', 'jpeg', 'png', 'bmp']:

                image = DataAugmentation.openImage(img_path)

                random_func = np.random.randint(0, 3)

                flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
                func=funcLists[random_func]

                if func == 'flip':
                    new_image_arr = DataAugmentation.randomFlip(image)
                    
                if func == 'rotation':
                    new_image_arr = DataAugmentation.rotate_image(image)

                elif func == 'crop':
                    new_image_arr = DataAugmentation.randomCrop(image)

                else:
                    new_image_arr = funcMap[func](image)
                
                new_image = Image.fromarray(new_image_arr, mode='RGB')
                DataAugmentation.saveImage(new_image, saved_img_path)



if __name__ == '__main__':
    
    # aug_app1(times=1)

    # aug_app2(times=2)

    aug_app3(times=3)

    