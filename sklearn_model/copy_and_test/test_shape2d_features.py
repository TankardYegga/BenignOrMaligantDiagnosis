# -*- encoding: utf-8 -*-
"""
@File    : test_shape2d_features.py
@Time    : 11/4/2021 7:57 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""

from radiomics import firstorder, featureextractor
from radiomics import shape2D
import cv2
import SimpleITK as sitk

img_path = r'D:\AllExploreDownloads\IDM\Data\CASE1-25\CASE1-25\Benign\CASE1\1CC.jpg'
mask_path = r'D:\AllExploreDownloads\IDM\Data\CASE1-25\CASE1-25\Benign\CASE1\1CC_output.jpg'
img = sitk.ReadImage(img_path, sitk.sitkInt8)
mask = sitk.ReadImage(mask_path, sitk.sitkInt8)
# img = sitk.ReadImage(img_path)
# mask = sitk.ReadImage(mask_path)

# img_arr = cv2.imread(img_path)
# mask_arr = cv2.imread(mask_path)
# img = sitk.GetImageFromArray(img_arr)
# mask = sitk.GetImageFromArray(mask_arr)

settings = {'force2D': True}
firstOrderFeatures = firstorder.RadiomicsFirstOrder(img, mask, **settings)
# firstOrderFeatures = firstorder.RadiomicsFirstOrder(img, mask)
firstOrderFeatures.enableAllFeatures() # On the feature class level, all features are disabled by default.
firstOrderFeatures.execute()
print(dir(firstOrderFeatures))
print(firstOrderFeatures.getEnergyFeatureValue())
print(firstOrderFeatures.getTotalEnergyFeatureValue())
print(firstOrderFeatures.getEntropyFeatureValue())
# firstOrderFeatures.calculateFeatures()
# for (key,val) in six.iteritems(firstOrderFeatures.featureValues):
# print("\t%s: %s" % (key, val))


shape2DFeatures = shape2D.RadiomicsShape2D(img, mask, **settings)
print(dir(shape2DFeatures))
shape2DFeatures.execute()
shape2DFeatures.enableAllFeatures()
# print(shape2DFeatures.getMeshSurfaceFeatureValue())
print(shape2DFeatures.getElongationFeatureValue())
print(shape2DFeatures.getMeshSurfaceFeatureValue())
print(shape2DFeatures.getMajorAxisLengthFeatureValue())
print(shape2DFeatures.getMaximumDiameterFeatureValue())
print(shape2DFeatures.getSphericalDisproportionFeatureValue())

settings = {
        'binWidth': 20,
        'sigma': [1, 2, 3],
        'verbose': True,
        #'label': 255,
        'force2D': True
    }
extractor = featureextractor.RadiomicsFeatureExtractor(additionInfo=True, **settings)
extractor.enableImageTypeByName('LoG')     # 即使设置了LoG，但是没有设置sigma的值不会计算该特征
extractor.enableImageTypeByName('LBP2D')    # 无需其他参数  93
extractor.enableImageTypeByName('Wavelet')  # 默认情况下就有LL,LH, HL,HH四种滤波 4 * 93
extractor.enableFeatureClassByName('shape2D') # 鼻血得设置参数
result = extractor.execute(img, mask)
print(result['original_shape2D_Elongation'])
print(result['original_shape2D_MeshSurface'])
print(result['original_shape2D_MajorAxisLength'])
print(result['original_shape2D_MaximumDiameter'])
print(result.keys())
print("len result1", len(result))


pop_keys = list(result.keys())[:22]
for key in list(result.keys()):
    if key in pop_keys:
        result.pop(key)

supplemented_result = dict()
supplemented_result.update({'original_shape2D_SphericalDisproportion':
                                shape2DFeatures.getSphericalDisproportionFeatureValue()})
for key, val in result.items():
    supplemented_result.update({key:val})

print(len(supplemented_result))
