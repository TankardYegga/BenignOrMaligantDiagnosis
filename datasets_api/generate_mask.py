# -*- encoding: utf-8 -*-
"""
@File    : generate_mask.py
@Time    : 11/3/2021 10:21 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
获取原始数据文件夹Data下每张图片的ROI MASK
"""

import cv2
import numpy as np
import os
from __init__ import global_var


def contour_point_num(contours):
    if len(contours) == 0:
        return 0
    point_num = 0
    for i in range(len(contours)):
        point_num += contours[i].shape[0]
    return point_num


def generate_mask_by_contour(img_path, save_path=None):

    # src = cv2.imread(img_path)
    # src = cv2.imread("8LMLO.jpg")
    # src = cv2.imread("3CC.jpg")

    # cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("input", src)
    src = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    green_low_hsv = np.array([35,43,46])
    green_high_hsv = np.array([77,255,255])
    yellow_low_hsv = np.array([26,43,46])
    yellow_high_hsv = np.array([34,255,255])
    red_low_hsv = np.array([0, 43, 46])
    red_high_hsv = np.array([10, 255, 255])

    ROI = np.zeros(src.shape, np.uint8)

    is_red = is_green = is_yellow = False
    is_green = True

    green_mask = cv2.inRange(hsv, lowerb=green_low_hsv, upperb=green_high_hsv)
    green_contours, green_hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, # EXTERNAL选择最外框
                                               cv2.CHAIN_APPROX_SIMPLE)

    yellow_mask = cv2.inRange(hsv, lowerb=yellow_low_hsv, upperb=yellow_high_hsv)
    yellow_contours, yellow_hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,  # EXTERNAL选择最外框
                                           cv2.CHAIN_APPROX_SIMPLE)

    red_mask = cv2.inRange(hsv, lowerb=red_low_hsv, upperb=red_high_hsv)
    red_contours, red_hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,  # EXTERNAL选择最外框
                                           cv2.CHAIN_APPROX_SIMPLE)

    yellow_count_num = contour_point_num(yellow_contours)
    red_count_num = contour_point_num(red_contours)
    green_count_num = contour_point_num(green_contours)
    longest_contours = green_contours if green_count_num > yellow_count_num else yellow_contours

    longest_contours = red_contours if red_count_num > contour_point_num(longest_contours) else longest_contours

    contours = longest_contours

    if len(contours) == 0:
        is_red = False
        print(img_path + " 图片中不存在绿色框也不存在黄色框也不存在红色框")
        return

    found_contours_first = contours[0]
    for i in range(1, len(contours)):
        found_contours_first = np.concatenate((found_contours_first, contours[i]), axis=0)

    rect = cv2.minAreaRect(found_contours_first)  # 找到最小外接矩形，该矩形可能有方向
    box = cv2.boxPoints(rect)
    cv2.drawContours(ROI, [box.astype(int)], -1, (255, 255, 255), -1)

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = save_path if save_path else os.path.dirname(img_path)
    output_path = os.path.join(save_path, img_name + '_output.jpg')
    print(output_path)
    # cv2.imwrite(output_path, ROI) 含有中文无法使用这种方式

    cv2_write_info = cv2.imencode('.jpg', ROI)
    cv2_write_info[1].tofile(output_path)
    #
    # cv2.imshow("test", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def find_all_required_files(file_extensions_list, source_path, file_path_list):
    for file_or_dir in os.listdir(source_path):
        cur_path = os.path.join(source_path, file_or_dir)
        if os.path.isfile(cur_path):
            cur_file_ext = os.path.splitext(os.path.basename(cur_path))
            # 如果文件没有后缀名，不再处理
            if len(cur_file_ext) < 2:
                continue
            if cur_file_ext[1] in file_extensions_list:
                file_path_list.append(cur_path)
                print(cur_path)
        elif os.path.isdir(cur_path):
            find_all_required_files(file_extensions_list, cur_path, file_path_list)
        else:
            continue


# 首先应该遍历目录，依次找到每张图片
# 然后对每张图片进行单独处理
def process_all_imgs(source_path):
    file_extensions_list = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG', '.bmp']
    file_path_list = []
    find_all_required_files(file_extensions_list, source_path, file_path_list)
    for single_img in file_path_list:
        if 'output' in single_img:
            continue
        generate_mask_by_contour(single_img)
    # generate_mask_by_contour(file_path_list[18])


if __name__ == '__main__':
    # source_path = 'D:\\AllExploreDownloads\\IDM\\Data'
    # process_all_imgs(source_path)

    single_img_path = global_var.base_data_api_prefix + '/test_data/M5LCC-1.jpg'
    generate_mask_by_contour(single_img_path)



