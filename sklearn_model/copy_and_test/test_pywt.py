# -*- encoding: utf-8 -*-
"""
@File    : test_pywt.py.py
@Time    : 11/23/2021 12:56 AM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

original_data = cv2.imread(r'D:\AllExploreDownloads\IDM\ExtraCode\test\B4RMLO.bmp',0)
print(original_data.shape)

data_decomposed = []
coefficient = []
num_of_decomposed = 3
data_decomposed.append(original_data)


def plt_wl_decomposition( ll,  lh,  hl,  hh):
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    fig = plt.figure()
    for i, data_decompostion in enumerate([ll, lh, hl, hh]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(data_decompostion, interpolation='nearest', cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i], fontsize=12)
    coeffs = ll, (lh, hl, hh)
    reconstructed = pywt.idwt2(coeffs, 'haar')
    plt.imshow(reconstructed, interpolation='nearest', cmap=plt.cm.gray)
    plt.show()


for i in range(num_of_decomposed):
    last_data = data_decomposed[-1]
    coeffs = pywt.dwt2(last_data, 'haar')
    ll, (lh, hl, hh) = coeffs
    data_decomposed.append(ll)
    coefficient.append(coeffs)
    plt_wl_decomposition(ll, lh, hl, hh)
