import cv2
import numpy as np
import os

import sys
sys.path.append('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/sklearn_model/')

from __init__ import global_var

def generate_black_whole_mask(img_dir, save_dir):
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img_arr = cv2.imread(img_path)

        print("img shape:", img_arr.shape)
        h, w = img_arr.shape[0], img_arr.shape[1]
        print(f"h:{h} w:{w}")

        black_whole_mask = np.ones((h, w), dtype=np.uint8) * 255
        black_whole_mask = cv2.cvtColor(black_whole_mask, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(save_dir, img_name[:-4] + '_mask.jpg'), black_whole_mask)
    

if __name__ == "__main__":
    img_dir = global_var.base_data_prefix + '/roi'
    save_dir = global_var.base_data_prefix + '/roi_mask'
    generate_black_whole_mask(img_dir, save_dir)

