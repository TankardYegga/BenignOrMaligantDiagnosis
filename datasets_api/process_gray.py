from concurrent.futures import thread
import cv2
import os
from cv2 import threshold
import numpy as np



base_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/roi2'
threshold=165
save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/train_part_of_original_data/changed_roi' + str(threshold)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


min_gray_list = []
max_gray_list = []


for img_full_name in os.listdir(base_dir):
    img_path = os.path.join(base_dir, img_full_name)
    img_arr = cv2.imread(img_path, 0)
    changed_img_arr = np.zeros_like(img_arr)
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if img_arr[i][j] >= threshold:
                changed_img_arr[i][j] = img_arr[i][j]
    save_path = os.path.join(save_dir, img_full_name)
    if np.count_nonzero(changed_img_arr) <= 10:
        print(img_full_name + ' skipped')
        continue
    changed_img_arr = cv2.cvtColor(changed_img_arr, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_path, changed_img_arr)
    min_gray_list.append(np.min(img_arr))
    max_gray_list.append(np.max(img_arr))

# print('min gray list:', min_gray_list)
# print('max gray list:', max_gray_list)



