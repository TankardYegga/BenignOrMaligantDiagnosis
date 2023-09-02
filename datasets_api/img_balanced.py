import cv2
import numpy as np
import os

from torch import eq


def balance_img(img_dir, save_dir):
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        save_path = os.path.join(save_dir, img)
        img = cv2.imread(img_path,0)
        equ = cv2.equalizeHist(img)
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(save_path, equ)


if __name__ == "__main__":
    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/roi/'
    # save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/roi_balanced/'
    # balance_img(img_dir, save_dir)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/roi/'
    # save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test/threshold_roi/'
    # balance_img(img_dir, save_dir)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/train/roi/'
    # save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/train/threshold_roi/'
    # balance_img(img_dir, save_dir)

    # img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/roi/'
    # save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/val/threshold_roi/'
    # balance_img(img_dir, save_dir)

    img_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test2/roi/'
    save_dir = '/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/datasets/data_corrected2/fixed_split_4/test2/threshold_roi/'
    balance_img(img_dir, save_dir)


