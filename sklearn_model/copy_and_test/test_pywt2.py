# -*- encoding: utf-8 -*-
"""
@File    : test_pywt2.py
@Time    : 11/23/2021 8:33 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

original_data = cv2.imread(r'D:\AllExploreDownloads\IDM\ExtraCode\test\B4RMLO.bmp', 0)
original_data = cv2.imread(r'D:\AllExploreDownloads\IDM\ExtraCode\test\M35LCC.bmp', 0)
print(original_data.shape)
plt.figure(figsize=(30, 30))
plt.imshow(original_data, interpolation='nearest', cmap=plt.cm.gray)
plt.show()

coeffs = pywt.wavedec2(original_data, 'haar', level=3)
cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

cAH3 = np.concatenate([cA3, cH3 + 1020], axis=1)
cVD3 = np.concatenate([cV3 + 1020, cD3 + 1020], axis=1)
c3 = np.concatenate([cAH3, cVD3], axis=0)

cAH2 = np.concatenate([c3, cH2 + 510], axis=1)
cVD2 = np.concatenate([cV2 + 510, cD2 + 510], axis=1)
c2 = np.concatenate([cAH2, cVD2], axis=0)

cAH1 = np.concatenate([c2, cH1 + 255], axis=1)
cVD1 = np.concatenate([cV1 + 255, cD1 + 255], axis=1)
c1 = np.concatenate([cAH1, cVD1], axis=0)

plt.figure(figsize=(30, 30))
plt.imshow(c1, interpolation='nearest', cmap=plt.cm.gray)
plt.show()


reconstructed_img = pywt.waverec2(coeffs, 'haar')
print('reconstructed_img shape:', reconstructed_img.shape)
print(reconstructed_img.dtype)
print(np.max(reconstructed_img))
print(np.min(reconstructed_img))
plt.figure(figsize=(30, 30))
plt.imshow(reconstructed_img, interpolation='nearest', cmap=plt.cm.gray)
plt.show()

coeffs[0] = np.zeros_like(coeffs[0])
coeffs_2 = list(coeffs[-2])
coeffs_1 = list(coeffs[-1])
coeffs_2[-1] = np.zeros_like(coeffs_2[-1])
coeffs_1[-1] = np.zeros_like(coeffs_1[-1])
coeffs[-1] = tuple(coeffs_1)
coeffs[-2] = tuple(coeffs_2)


reconstructed_img = pywt.waverec2(coeffs, 'haar')
print('reconstructed_img shape:', reconstructed_img.shape)
print(reconstructed_img.dtype)
print(np.max(reconstructed_img))
print(np.min(reconstructed_img))
plt.figure(figsize=(30, 30))
plt.imshow(reconstructed_img, interpolation='nearest', cmap=plt.cm.gray)
plt.show()

another_img = np.zeros_like(reconstructed_img)
for i in range(reconstructed_img.shape[0]):
    for j in range(reconstructed_img.shape[-1]):
        if reconstructed_img[i][j] <= -20:
            another_img[i][j] = 255
cv2.imwrite('test_seg0.png', another_img)
plt.figure(figsize=(30, 30))
plt.imshow(another_img, interpolation='nearest', cmap=plt.cm.gray)
plt.show()

# reconstructed_img = cv2.cvtColor(reconstructed_img, cv2.COLOR_GRAY2BGR)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
img = cv2.morphologyEx(another_img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('test_seg.png', img)
plt.figure(figsize=(30, 30))
plt.imshow(img, interpolation='nearest',)
plt.show()