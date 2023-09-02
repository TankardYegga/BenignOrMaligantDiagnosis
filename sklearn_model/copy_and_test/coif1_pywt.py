import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os


def get_coif1_img(img_dir='/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/sklearn_model/copy_and_test/'):
    img_full_name = 'B7RCC.jpg'
    # img_full_name = 'M109LMLO.jpg'

    img_name = img_full_name.split('.')[0]
    img_arr = cv2.imread(os.path.join(img_dir, img_full_name))
    print(img_arr.shape)

    ca, (cb, cc, cd) = pywt.dwt2(img_arr, 'coif1')

    cb = np.zeros_like(cb)
    cc = np.zeros_like(cc)
    cd = np.zeros_like(cd)
    coefs = ca, (cb, cc, cd)
    reconstructed_img = pywt.idwt2(coefs, 'coif1')
    cv2.imwrite(os.path.join(img_dir, img_name + "_re.jpg"), reconstructed_img)
    
    # img_ll_arr = np.uint8(ca/np.max(ca) * 255)
    # img_ll_arr = cv2.resize(img_ll_arr, (256, 256))
    # cv2.imwrite(os.path.join(img_dir, img_name + '_ll.jpg'), img_ll_arr)

    # img_lr_arr = np.uint8(cb/np.max(cb) * 255)
    # img_lr_arr = cv2.resize(img_lr_arr, (256, 256))
    # cv2.imwrite(os.path.join(img_dir, img_name + '_lr.jpg'), img_lr_arr)

    # img_rl_arr = np.uint8(cc/np.max(cc) * 255)
    # img_rl_arr = cv2.resize(img_rl_arr, (256, 256))
    # cv2.imwrite(os.path.join(img_dir, img_name + '_rl.jpg'), img_rl_arr)

    # img_rr_arr = np.uint8(cd/np.max(cd) * 255)
    # img_rr_arr = cv2.resize(img_rr_arr, (256, 256))
    # cv2.imwrite(os.path.join(img_dir, img_name + '_rr.jpg'), img_rr_arr)


def get_topo_mask(img_dir='/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/sklearn_model/copy_and_test/'):
    img_full_name = 'M109LMLO.jpg'
    img_name = img_full_name.split('.')[0]
    gray_img_arr = cv2.imread(os.path.join(img_dir, img_full_name))
    print(gray_img_arr.shape)

    if len(gray_img_arr.shape) == 3:
        gray_img_arr = cv2.cvtColor(gray_img_arr, cv2.COLOR_RGB2GRAY)
    assert len(gray_img_arr.shape) == 2

    coeffs = pywt.wavedec2(gray_img_arr, 'haar', level=3)
    cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    coeffs[0] = np.zeros_like(coeffs[0])
    coeffs_2 = list(coeffs[-2])
    coeffs_1 = list(coeffs[-1])
    coeffs_2[-1] = np.zeros_like(coeffs_2[-1])
    coeffs_1[-1] = np.zeros_like(coeffs_1[-1])
    coeffs[-1] = tuple(coeffs_1)
    coeffs[-2] = tuple(coeffs_2)
    reconstructed_img = pywt.waverec2(coeffs, 'haar')
    cv2.imwrite(os.path.join(img_dir, img_name + '_haar.jpg'), reconstructed_img)
    # cv2.imwrite('topo_color.png', topo_mask_img)
    
get_coif1_img()
# get_topo_mask()