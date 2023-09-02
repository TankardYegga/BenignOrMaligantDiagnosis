#-*- coding:utf-8 -*-
from pydoc import doc
import cv2
import numpy as np
import os
from tqdm import tqdm
from __init__ import global_var

 
def order_points(pts):
    ''' sort rectangle points by clockwise '''
    sort_x = pts[np.argsort(pts[:, 0]), :]
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    # Left sort
    Left = Left[np.argsort(Left[:, 1])[::-1], :]
    # Right sort
    Right = Right[np.argsort(Right[:, 1]), :]
    return np.concatenate((Left, Right), axis=0)
 
 
def get_doc_area(img_path, mask_path):

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for j, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 0:
            continue
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = np.squeeze(approx)
        #print('approx_len:{}'.format(len(approx)))
        if len(approx) != 4:
            continue
        arbitrary_points = np.float32(order_points(approx))
        #print(arbitrary_points)
        h, w = arbitrary_points[0][1] - arbitrary_points[1][1], \
                arbitrary_points[3][0] - arbitrary_points[0][0]
        rectangle = np.float32([[0, h], [0, 0], [w, 0], [w, h]])

        M = cv2.getPerspectiveTransform(arbitrary_points, rectangle)
        h = int(h)
        w = int(w)
        doc_dst = cv2.warpPerspective(img, M, (w, h))
        mask_dst = cv2.warpPerspective(mask, M, (w, h))

        padding = -5
        h, w = doc_dst.shape[:2]
        doc_dst = doc_dst[-padding:h+padding, -padding:w+padding]
        cv2.imwrite(global_var.base_data_api_prefix + "/test_data/M5LCC-1_doc.jpg", doc_dst)

        # cv2.imshow('doc_dst', doc_dst)
        # cv2.imshow('mask_dst', mask_dst)
        # cv2.waitKey(0)


if __name__=='__main__':
    img_path =  global_var.base_data_api_prefix + '/test_data/M5LCC-1.jpg'
    mask_path = global_var.base_data_api_prefix + '/test_data/M5LCC-1_output.jpg'
    get_doc_area(img_path, mask_path)

    
