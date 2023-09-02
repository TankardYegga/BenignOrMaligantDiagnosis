# -*- encoding: utf-8 -*-
"""
@File    : feature_selection.py
@Time    : 11/3/2021 10:20 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
from cmath import pi
import sys
sys.path.append('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis')

import math
from tkinter.messagebox import NO
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from scipy.stats import pearsonr
import statsmodels.api as sm
import global_var
import pickle

def count_different_feat_num(feats_list):
    shape_feats_keys = ['shape2D']
    texture_feats_keys = ['firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm']
    topo_feats_keys = ['Degree', 'Vertex', 'Subgraph', 'Component', 'Average', 'Points', 'Diameter']

    shape_feats = []
    texture_feats = []
    topo_feats = []
    unrecognized_feats = []
    for feat in feats_list:
        is_shape = False
        is_texture = False
        is_topo = False
        for key in shape_feats_keys:
            if key in feat:
                shape_feats.append(feat)
                is_shape = True
                break
        if is_shape:
            continue

        for key in texture_feats_keys:
            if key in feat:
                is_texture = True
                texture_feats.append(feat)
                break
        if is_texture:
            continue

        for key in topo_feats_keys:
            if key in feat:
                # print('topo feats:', feat)
                is_topo = True
                topo_feats.append(feat)
                break
        if is_topo:
            continue
        
        print('unrecognized_feats:', feat)
        unrecognized_feats.append(feat)

    print("shape_feats:", len(shape_feats))
    print("texture_feats:", len(texture_feats))
    print("topo_feats:", len(topo_feats))
    print("unrecognized_feats", len(unrecognized_feats))
    return shape_feats, texture_feats, topo_feats, unrecognized_feats


def z_score_filter(csv_file, feat_filtering_threshold = 20, z=3, eps=1e-9):
    features = pd.read_csv(csv_file)
    feat_values = features.values
    print("feat_values type:", feat_values.dtype)

    """
    打印原始特征信息
    """
    original_feats = list(features.columns)[2:]
    shape_feats, texture_feats, \
    topo_feats, unrecognized_feats = count_different_feat_num(original_feats)
    print("shape feats length:", len(shape_feats))
    print("texture feats length:", len(texture_feats))
    print("topo feats length:", len(topo_feats))
    print("unrecognized feats length:", len(unrecognized_feats))

    """
    利用Z分法删除特征
    """
    feat_info = feat_values[:, :2]
    feat_data = feat_values[:, 2:]
    feat_data = np.asarray(feat_data, dtype=np.float64)
    print("after trans feat_values type:", feat_data.dtype)

    mean_feat_dict = {}
    mean_feat_data = np.mean(feat_data, axis=0)
    for i in range(len(original_feats)):
        mean_feat_dict[original_feats[i]] = mean_feat_data[i]
    print('mean_feat_dict:\n',mean_feat_dict)
    with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict.pkl', "wb") as f:
        pickle.dump(mean_feat_dict, f, pickle.HIGHEST_PROTOCOL)
    mean_feat_data = mean_feat_data[np.newaxis, :]
    mean_feat_data = np.tile(mean_feat_data, (feat_data.shape[0], 1))

    std_feat_dict = {}
    std_feat_data = np.std(feat_data, axis=0)
    for i in range(len(original_feats)):
        std_feat_dict[original_feats[i]] = std_feat_data[i]
    print('std_feat_dict:\n', std_feat_dict)
    with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict.pkl', "wb") as f:
        pickle.dump(std_feat_dict, f, pickle.HIGHEST_PROTOCOL)
    std_feat_data = std_feat_data[np.newaxis, :]
    std_feat_data = np.tile(std_feat_data, (feat_data.shape[0], 1))

    # z_scored_feat_data = np.subtract(feat_data, mean_feat_data) / ( std_feat_data + eps )
    z_scored_feat_data = (feat_data - mean_feat_data) / ( std_feat_data + eps )

    z_scored_feat_data_df = pd.DataFrame(z_scored_feat_data)
    z_scored_feat_data_df.to_csv(global_var.base_sklearn_prefix + '/features_saved/merged_features/z_score_diy.csv', index=True)

    # sys.exit(0)

    to_be_removed_feats = []
    feat_columns = list(features.columns)

    for col in range(2, z_scored_feat_data.shape[-1]):
        feat = z_scored_feat_data[:, col]
        count = 0
        for value_idx in range(len(feat)):
            value = feat[value_idx]
            if np.isnan(value) or (value < -z or value > z):
                # print('error value:', value)
                count += 1
        if count > feat_filtering_threshold:
            to_be_removed_feats.append(feat_columns[col])

    # print(to_be_removed_feats)
    # print(len(to_be_removed_feats))

    """
    打印移除的特征信息
    """
    shape_feats_removed, texture_feats_removed, \
        topo_feats_removed, unrecognized_feats_removed = count_different_feat_num(to_be_removed_feats)

    print("Removed shape feats length:", len(shape_feats_removed))
    print("Removed texture feats length:", len(texture_feats_removed))
    print("Removed topo feats length:", len(topo_feats_removed))
    print("Removed unrecognized feats length:", len(unrecognized_feats_removed))

    return to_be_removed_feats


def save_filtered_features(original_csv_file, save_csv_path, to_be_removed_feats, is_standardized = True):
    """
    将删除且Z分化的特征信息保存成新文件
    """
    # 先从原始csv文件中利用pandas读取
    original_feats_df = pd.read_csv(original_csv_file)

    # 移除掉特征
    filtered_feats_df = original_feats_df.drop(to_be_removed_feats, axis=1)

    std_feat_dict = {}
    mean_feat_dict = {}

    # 进行标准化
    if is_standardized:
        standard_normalization_scaler = lambda x: (x - np.mean(x)) / ( np.std(x) + 1e-9 )
        # standard_normalization_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
        to_be_scaled_columns = filtered_feats_df.columns.tolist()
        to_be_scaled_columns.remove('image')
        to_be_scaled_columns.remove('label')
        df_array = filtered_feats_df.values[:, 2:]
        df_array = np.asarray(df_array, np.float64)

        for to_be_scaled_column in to_be_scaled_columns:
            col_num = to_be_scaled_columns.index(to_be_scaled_column)
            mean_feat_dict[to_be_scaled_column] = np.mean(df_array[:, col_num])
            std_feat_dict[to_be_scaled_column] = np.std(df_array[:, col_num])
            # x = filtered_feats_df[to_be_scaled_column]
            # filtered_feats_df[to_be_scaled_column] = (x - np.mean(x)) / np.std(x)
            filtered_feats_df[[to_be_scaled_column]] = filtered_feats_df[[to_be_scaled_column]].apply(standard_normalization_scaler)
        # 进行缺失值和异常值的处理

    with open(global_var.base_feature_prefix + '/merged_features/mean_feats_dict2.pkl', "wb") as f:
        pickle.dump(mean_feat_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(global_var.base_feature_prefix + '/merged_features/std_feats_dict2.pkl', "wb") as f:
        pickle.dump(std_feat_dict, f, pickle.HIGHEST_PROTOCOL)
    
    # 保存文件
    filtered_feats_df.to_csv(save_csv_path, index=True)


def z_score_by_df_apply(csv_file, save_path='./merged_features/z_score_apply.csv'):
    df = pd.read_csv(csv_file)
    standard_normalization_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    df_columns = df.columns.tolist()
    df_columns.remove('image')
    df_columns.remove('label')
    df[df_columns] = df[df_columns].apply(standard_normalization_scaler)
    df.to_csv(save_path, index=True)


def z_score_by_standard_scaler(csv_file, save_path='./merged_features/z_score_standard_scaler.csv'):
    df = pd.read_csv(csv_file)
    feat_mat = df.values[:, 2:]
    feat_mat = np.asarray(feat_mat, dtype=np.float64)
    feat_mat = StandardScaler().fit_transform(feat_mat)
    print("type :", type(feat_mat))
    z_scored_feat_arr = pd.DataFrame(feat_mat)
    z_scored_feat_arr.to_csv(save_path, index=True)


def rectify_missing_and_abnormal_value(csv_file, rectified_save_path, z = 3):
    df = pd.read_csv(csv_file)
    df_columns_list = df.columns.tolist()

    """因为df.to_csv时会默认加上一列表示行号，所以读取使用这种方法保存的文件时要注意第一列是否会干扰操作"""
    df.drop(axis=1, columns=df_columns_list[0], inplace=True)
    df_columns_list = df.columns.tolist()
    df_columns_list.remove('image')
    df_columns_list.remove('label')
    print("dir df:", dir(df))
    # 因为多增加了行号列，所以需要从3而不是2开始索引
    feat_mat = np.asarray(df.values[:, 2:], dtype=np.float64)

    # 处理缺失值，即使用平均值来填充
    imputation_transformer = SimpleImputer(missing_values=np.nan, strategy='mean')
    feat_mat = imputation_transformer.fit_transform(feat_mat)

    # 处理异常值，即对于计算得到的Z分数在-3到3之外的，将其强制转化到正常范围之内
    positive_sign = lambda x: 1 if x >= 0 else -1
    abnormal_value_rectifier = lambda x: positive_sign(x) * 3 if abs(x) > 3 else x
    for col_idx in range(feat_mat.shape[-1]):
        single_feat = list(feat_mat[:, col_idx])
        single_feat_rectified = list(map(abnormal_value_rectifier, single_feat))
        single_feat_rectified_series = pd.Series(single_feat)
        df[df_columns_list[col_idx]] = single_feat_rectified

    df.to_csv(rectified_save_path, index=True)


def anova_filtered_features(df_feat_data, df_label_data, is_remaining_list):
    """对原始数据进行分组，分别获得良性和恶性这两个组别的相应特征值"""
    valid_feats_indexes = []
    for i in range(len(is_remaining_list)):
        if is_remaining_list[i]:
            valid_feats_indexes.append(i)
    print("len of remaining feats before anova filtering:", len(is_remaining_list.nonzero()[0]))

    valid_feats_indexes = np.asarray(valid_feats_indexes)
    df_feat_data = df_feat_data[:, valid_feats_indexes]

    benign_feats_indexes = []
    maligant_feats_indexes = []
    for i in range(len(df_label_data)):
        if df_label_data[i] == 'Benign':
            benign_feats_indexes.append(i)
        elif df_label_data[i] == 'Malignant':
            maligant_feats_indexes.append(i)
        else:
            continue

    benign_feats_indexes = np.asarray(benign_feats_indexes)     # 210 Benign
    maligant_feats_indexes = np.asarray(maligant_feats_indexes) # 218 Malignant
    benign_feats_mat = df_feat_data[benign_feats_indexes, :]
    maligant_feats_mat = df_feat_data[maligant_feats_indexes, :]
    feats_f, feats_p = stats.f_oneway(benign_feats_mat, maligant_feats_mat)
    feats_p_cp = feats_p.copy()
    feats_p_cp.sort()
    plt.plot(feats_p_cp)
    plt.savefig(global_var.base_sklearn_prefix + "/extra_saved/anova_filter.jpg", dpi=300, pad_inches=0)
    plt.show()

    filtered_valid_feats_idxes = feats_p < 0.05
    remaining_feats_indexes = valid_feats_indexes[filtered_valid_feats_idxes]
    is_remaining_list = np.asarray([1 if i in remaining_feats_indexes else 0 for i in range(len(is_remaining_list))])
    print("len of remaining feats after anova filtering:", len(is_remaining_list.nonzero()[0]))
    return is_remaining_list


def pearson(vector1, vector2):
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den


def pearson_filtered_features(df_feat_data, is_remaining_list):
    valid_feats_indexes = []
    for i in range(len(is_remaining_list)):
        if is_remaining_list[i]:
            valid_feats_indexes.append(i)
    print("len of remaining feats before pearson filtering:", len(is_remaining_list.nonzero()[0]))

    valid_feats_indexes = np.asarray(valid_feats_indexes)
    df_feat_data = df_feat_data[:, valid_feats_indexes]

    """计算皮尔森矩阵"""
    num_of_cases = df_feat_data.shape[0]
    num_of_feats = df_feat_data.shape[-1]
    pearson_arr = np.zeros(shape=(num_of_feats, num_of_feats), dtype=np.float64)
    for i in range(num_of_feats):
        for j in range(num_of_feats):
            # pearson_arr[i][j] = pearson(df_feat_data[:, i], df_feat_data[:, j])
            pearson_arr[i][j] = pearsonr(df_feat_data[:, i], df_feat_data[:, j])[0]

    """显示皮尔森热力图"""
    plt.figure(figsize=(20, 20), facecolor='white')
    # seaborn.heatmap(pearson_arr, cmap='Blues')
    seaborn.heatmap(pearson_arr)
    plt.savefig(global_var.base_sklearn_prefix + '/extra_saved/pearson_filter.jpg')
    plt.show()


    """依据矩阵来筛选特征"""
    # 这里面会动态的改变数组的大小 那么遍历的时候肯定会出问题啊
    re_linear_feats_idxes = np.array([], dtype=np.uint8)
    pearson_filtered_feats_idxes = []
    # for i in range(num_of_feats - 1, -1, -1):
    #     pearson_filtered_feats_idxes.append(i)
    #     if  i == 0:
    #         continue
    #     if i not in re_linear_feats_idxes:
    #         max_linear = - sys.maxsize - 1
    #         max_relinear_idx = -1
    #         for j in range(0, i):
    #             if j != i and j not in re_linear_feats_idxes and pearson_arr[i, j] > max_linear:
    #                 max_linear = pearson_arr[i, j]
    #                 max_relinear_idx = j
    #         if max_relinear_idx != -1:
    #             re_linear_feats_idxes = np.append(re_linear_feats_idxes, max_relinear_idx)

    # for i in range(num_of_feats - 1, -1, -1):
    #     pearson_filtered_feats_idxes.append(i)
    #     if i == 0:
    #         continue
    #     if i not in re_linear_feats_idxes:
    #         max_linear = - sys.maxsize - 1
    #         max_relinear_idx = -1
    #         for j in range(num_of_feats):
    #             if j != i and j not in re_linear_feats_idxes and pearson_arr[i, j] > max_linear:
    #                 max_linear = pearson_arr[i, j]
    #                 max_relinear_idx = j
    #         if max_relinear_idx != -1:
    #             re_linear_feats_idxes = np.append(re_linear_feats_idxes, max_relinear_idx)

    for i in range(num_of_feats - 1, -1, -1):
        pearson_filtered_feats_idxes.append(i)
        if i not in re_linear_feats_idxes:
            max_linear = 0.5
            max_relinear_idx = -1
            for j in range(num_of_feats):
                if j != i and j not in re_linear_feats_idxes and abs(pearson_arr[i, j]) >= max_linear:
                    max_linear = abs(pearson_arr[i, j])
                    max_relinear_idx = j
            if max_relinear_idx != -1:
                re_linear_feats_idxes = np.append(re_linear_feats_idxes, max_relinear_idx)
        
        # if i == 0:
        #     break

    # for i in range(num_of_feats):
    #     pearson_filtered_feats_idxes.append(i)
    #     if  i == num_of_feats - 1:
    #         continue
    #     if i not in re_linear_feats_idxes:
    #         max_linear = - sys.maxsize - 1
    #         max_relinear_idx = -1
    #         for j in range(i+1, num_of_feats):
    #             if j != i and j not in re_linear_feats_idxes and pearson_arr[i, j] > max_linear:
    #                 max_linear = pearson_arr[i, j]
    #                 max_relinear_idx = j
    #         if max_relinear_idx != -1:
    #             re_linear_feats_idxes = np.append(re_linear_feats_idxes, max_relinear_idx)

    # for i in range(num_of_feats):
    #     pearson_filtered_feats_idxes.append(i)
    #     if i == num_of_feats - 1:
    #         continue
    #     if i not in re_linear_feats_idxes:
    #         max_linear = - sys.maxsize - 1
    #         max_relinear_idx = -1
    #         for j in range(num_of_feats):
    #             if j != i and j not in re_linear_feats_idxes and pearson_arr[i, j] > max_linear:
    #                 max_linear = pearson_arr[i, j]
    #                 max_relinear_idx = j
    #         if max_relinear_idx != -1:
    #             re_linear_feats_idxes = np.append(re_linear_feats_idxes, max_relinear_idx)

    for idx in range(len(re_linear_feats_idxes)):
        try:
            pearson_filtered_feats_idxes.remove(re_linear_feats_idxes[idx])
            if len(pearson_filtered_feats_idxes) != len(np.unique(pearson_filtered_feats_idxes)):
                print('error', len(pearson_filtered_feats_idxes), ":", len(np.unique(pearson_filtered_feats_idxes)))
        except Exception as e:
            print('feat idx:', re_linear_feats_idxes[idx])
            print('idx:', idx)
            print(e)

    pearson_filtered_feats_idxes_cp = np.asarray(pearson_filtered_feats_idxes)
    remaining_feats_indexes = valid_feats_indexes[pearson_filtered_feats_idxes]

    is_remaining_list = np.asarray([1 if i in remaining_feats_indexes else 0 for i in range(len(is_remaining_list))])
    print("len of remaining feats after pearson filtering:", len(is_remaining_list.nonzero()[0]))

    return is_remaining_list


def ols_filtered_features(df_feat_data, df_label_data, df_columns, is_remaining_list, step = 1):
    valid_feats_indexes = []
    for i in range(len(is_remaining_list)):
        if is_remaining_list[i]:
            valid_feats_indexes.append(i)
    print("len of remaining feats before ols filtering:", len(is_remaining_list.nonzero()[0]))

    valid_feats_indexes = np.asarray(valid_feats_indexes)
    df_label_data = np.asarray([0 if label == 'Benign' else 1 for label in df_label_data])

    X = df_feat_data
    Y = df_label_data

    remaining_feats_original_idxex = valid_feats_indexes

    est = sm.OLS(Y, sm.add_constant(X[:, remaining_feats_original_idxex])).fit()
    # 注意这里返回值的长度，一般等于对应的特征的数目，但是因为这里使用add_constant所以多增加了1列，长度也会相应加上1
    feat_pvalues_arr = est.pvalues[1:]
    inapparent_feats_num = len(( feat_pvalues_arr >= 0.05).nonzero()[0])
    inapparent_feats_dict = dict(zip(np.where(feat_pvalues_arr >= 0.05)[0].tolist(),
                                    feat_pvalues_arr[feat_pvalues_arr >= 0.05].tolist()))
    # print("est pvalues:", est.pvalues)
    while inapparent_feats_num:
        inapparent_feats_idxes = []
        inapparent_feats_dict_list = sorted(inapparent_feats_dict.items(), key = lambda item: item[1], reverse=True)
        removed_feats_num = min(step, inapparent_feats_num)
        for i in range(removed_feats_num):
            inapparent_feats_idxes.append(inapparent_feats_dict_list[i][0])

        # 这里的问题是怎么把这些指定idx的位置给删除掉
        # 你把一个位置移除了，就会导致整个list的索引空间发生变化
        # 那只能做一份拷贝，利用移除值的方式来删除
        remaining_feats_original_idxex_cp = list(remaining_feats_original_idxex).copy()
        for idx in inapparent_feats_idxes:
            remaining_feats_original_idxex_cp.remove(remaining_feats_original_idxex[idx])
        remaining_feats_original_idxex = np.asarray(remaining_feats_original_idxex_cp)

        # 将这些特征移除掉再次计算est
        est = sm.OLS(Y, sm.add_constant(X[:, remaining_feats_original_idxex])).fit()
        feat_pvalues_arr = est.pvalues[1:]
        inapparent_feats_num = len((feat_pvalues_arr >= 0.05).nonzero()[0])
        inapparent_feats_dict = dict(zip(np.where(feat_pvalues_arr >= 0.05)[0].tolist(),
                                         feat_pvalues_arr[feat_pvalues_arr >= 0.05].tolist()))
        # print("est pvalues:", est.pvalues)

    is_remaining_list = np.asarray([1 if i in remaining_feats_original_idxex else 0 for i in range(len(is_remaining_list))])
    print("len of remaining feats after ols filtering:", len(is_remaining_list.nonzero()[0]))

    return is_remaining_list


def xgboost_rfe_filtered_features(df_feat_data, df_label_data, df_columns, is_remaining_list,
                                  n_features_to_select=25):
    valid_feats_indexes = []
    for i in range(len(is_remaining_list)):
        if is_remaining_list[i]:
            valid_feats_indexes.append(i)
    print("len of remaining feats before xgboost filtering:", len(is_remaining_list.nonzero()[0]))

    valid_feats_indexes = np.asarray(valid_feats_indexes)
    df_feat_data = df_feat_data[:, valid_feats_indexes]
    df_label_data = np.asarray([0 if label == 'Benign' else 1 for label in df_label_data])

    selector = RFE(estimator=XGBClassifier(use_label_encoder=False), n_features_to_select=n_features_to_select, step=1)
    selector.fit(df_feat_data, df_label_data)
    # res = selector.fit_transform(df_feat_data, df_label_data)  # 这个只会单纯地返回筛选后的特征,类型是numpy数组

    print("feats number before rfe:", len(selector.ranking_))
    print("ranking:", selector.ranking_)
    print("support:", selector.support_)
    print("features:", selector.n_features_)
    print("n_features_to_select:", selector.n_features_to_select)
    print("n_features_in_", selector.n_features_in_)
    print("n_features:",  (selector.ranking_ == 1).nonzero()[0])
    print("model feature importances:", selector.estimator_.feature_importances_)
    # print(dir(selector.estimator_))
    # print(dir(selector.estimator))

    # fig, axes = plt.subplots(1, 1, figsize=(20, 20))
    plt.figure(figsize=(60, 60))
    plt.title('xgboost_feats_importances' + str(selector.n_features_))
    plt.barh(np.asarray(df_columns)[valid_feats_indexes][selector.support_],
             selector.estimator_.feature_importances_,
             facecolor='blue', edgecolor='blue',
             alpha=0.5)
    plt.xlabel(u'Feature Importance Score', fontsize=20)
    plt.ylabel(u'Features', fontsize=20)
    plt.legend(loc='best')
    img_name = global_var.base_sklearn_prefix + '/extra_saved/xgboost_feats_importances' + str(selector.n_features_) + '.jpg'
    plt.savefig(img_name)
    plt.show()

    remaining_feats_indexes = valid_feats_indexes[selector.support_]
    is_remaining_list = np.asarray([1 if i in remaining_feats_indexes else 0 for i in range(len(is_remaining_list))])
    print("len of remaining feats after xgboost filtering:", len(is_remaining_list.nonzero()[0]))
    return is_remaining_list


def xgboost_rfecv_filtered_features(df_feat_data, df_label_data, df_columns, is_remaining_list):
    valid_feats_indexes = []
    for i in range(len(is_remaining_list)):
        if is_remaining_list[i]:
            valid_feats_indexes.append(i)
    print("len of remaining feats before xgboost filtering:", len(is_remaining_list.nonzero()[0]))

    valid_feats_indexes = np.asarray(valid_feats_indexes)
    df_feat_data = df_feat_data[:, valid_feats_indexes]
    df_label_data = np.asarray([0 if label == 'Benign' else 1 for label in df_label_data])

    model = XGBClassifier()
    selector = RFECV(estimator=model, step=1, cv=5)
    selector.fit(df_feat_data, df_label_data)
    res = selector.fit_transform(df_feat_data, df_label_data)  # 这个只会单纯地返回筛选后的特征,类型是numpy数组
    print("feats number before rfe:", len(selector.ranking_))
    print("ranking:", selector.ranking_)
    print("support:", selector.support_)
    print("features:", selector.n_features_)
    print("n_features_in_", selector.n_features_in_)
    print("n_features:",  (selector.ranking_ == 1).nonzero()[0])
    print("model feature importances:", selector.estimator_.feature_importances_)  # 单纯使用model会显示所有特征重要性的得分，因为不会像RFE一样对特征进行筛选

    plt.figure(figsize=(60, 60))
    plt.title('xgboost_feats_importances')
    plt.barh(np.asarray(df_columns)[valid_feats_indexes][selector.support_],
             selector.estimator_.feature_importances_,
             facecolor='blue', edgecolor='blue',
             alpha=0.5)
    plt.xlabel(u'Feature Importance Score', fontsize=20)
    plt.ylabel(u'Features', fontsize=20)
    plt.legend(loc='best')
    plt.savefig(global_var.base_sklearn_prefix + "/extra_saved/xgboost_cv_feats_importances.jpg")
    plt.show()

    remaining_feats_indexes = valid_feats_indexes[selector.support_]
    is_remaining_list = np.asarray([1 if i in remaining_feats_indexes else 0 for i in range(len(is_remaining_list))])
    print("len of remaining feats after xgboost filtering:", len(is_remaining_list.nonzero()[0]))
    return is_remaining_list


def filter_features_by_z_score(original_csv_file,  z_score_filtered_path = None):

    """通过Z分法找出需要被删除的特征（并没有进行归一化）"""
    to_be_removed_feats = z_score_filter(original_csv_file)

    """根据被删除的特征来保存标准化后的新的特征文件（进行了归一化）"""
    save_filtered_features(original_csv_file, z_score_filtered_path, to_be_removed_feats, is_standardized=True)

    """调用函数库来实现z分法，但是没有特征删除操作，只是为了测试下库函数的使用效果"""
    # z_score_by_df_apply(original_csv_file)
    # z_score_by_standard_scaler(original_csv_file)


def remove_repeated_and_constant_features(rectified_save_path, rectified_unrepeated_save_path=r'./merged_features/z_score_filtered_features_rectified_unrepeated.csv'):
    """读取特征文件"""

    df = pd.read_csv(rectified_save_path)
    df_columns_list = df.columns.tolist()
    df_feat_data = np.asarray(df.values[:, 3:], dtype=np.float64)
    df_label_data = np.asarray(df.values[:, 2])
    print(df_feat_data.shape)

    X = df_feat_data
    Y = df_label_data
    repeated_idx = set()
    repeated_pairs = []
    constant_idx = []
    for i in range(df_feat_data.shape[-1]):
        if np.all(X[:, i] == X[:, i][0]):
            constant_idx.append(i)
        for j in range(i + 1, df_feat_data.shape[-1]):
            if np.all(X[:, i] == X[:, j]):
                print("Shot!", i, " ", df_columns_list[i+3], ":",
                      j, " ", df_columns_list[j+3])
                repeated_idx.add(i)
                repeated_idx.add(j)
                if i < j:
                    repeated_pairs.append((i, j))
                else:
                    repeated_pairs.append((j, i))
    print("repeated_idx", repeated_idx)
    print("len: ", len(repeated_idx))

    removed_idx = []
    for repeated_pair in repeated_pairs:
        i = repeated_pair[0]
        j = repeated_pair[1]
        if i in repeated_idx and j in repeated_idx:
            repeated_idx.remove(j)
            removed_idx.append(j)
        elif i in repeated_idx:
            continue
        elif j in repeated_idx:
            continue
        else:
            continue
    print("after removing the repeated:", repeated_idx)
    print("len: ", len(repeated_idx))
    print("removed feat idx:", removed_idx)
    print("len of removed feat idx:", len(removed_idx))

    for i in repeated_idx:
        for j in repeated_idx:
            if j != i and np.all(X[:, i] == X[:, j]):
                print("Shot!", i, " ", ":",
                      j, " ")

    print('constant idx:', constant_idx)
    for idx in constant_idx:
        if idx not in removed_idx:
            removed_idx.append(idx)
    print("len of removed feat idx:", len(removed_idx))

    removed_cols = np.asarray([True if i in removed_idx else False for i in range(df_feat_data.shape[-1])])
    df_columns_list = np.asarray(df_columns_list)
    removed_feats_columns = df_columns_list[3:][removed_cols]
    for col in removed_feats_columns:
        print(col)
        # df.drop([col], axis=1, inplace=True)
        del df[col]

    print("*" * 100)

    print(df.shape)
    df.to_csv(rectified_unrepeated_save_path, index=False)


def filter_features(rectified_save_path, filtered_feats_save_path, n_features_to_select=25):
    """读取特征文件"""
    df = pd.read_csv(rectified_save_path)
    df_columns_list = df.columns.tolist()
    df_feat_data = np.asarray(df.values[:, 3:], dtype=np.float64)
    df_label_data = np.asarray(df.values[:, 2])
    is_remaining_list = np.asarray([1] * (len(df_columns_list) - 3))

    """使用ANOVA方法来筛选一部分特征"""
    is_remaining_list = anova_filtered_features(df_feat_data, df_label_data,
                                                             is_remaining_list)
    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                             dtype=bool)])

    """使用皮尔森系数来过滤特征"""
    is_remaining_list = pearson_filtered_features(df_feat_data, is_remaining_list)
    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                                                                        dtype=bool)])

    #
    """使用基于最小二乘法的后向特征筛选"""
    is_remaining_list = ols_filtered_features(df_feat_data, df_label_data, df_columns_list[3:], is_remaining_list)
    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                                                                        dtype=bool)])

    # """使用XGBoost来进行递归特征筛选"""
    is_remaining_list = xgboost_rfe_filtered_features(df_feat_data, df_label_data, df_columns_list[3:],
                                                      is_remaining_list, n_features_to_select=n_features_to_select)
    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                                                                        dtype=bool)])

    # is_remaining_list = xgboost_rfecv_filtered_features(df_feat_data, df_label_data, df_columns_list[3:],
    #                                                   is_remaining_list)
    # count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
    #                                                                     dtype=bool)])

    """保存最终过滤后的特征文件"""
    is_removed_list = [0] * len(is_remaining_list)
    for i in range(len(is_remaining_list)):
        is_removed_list[i] = 1 - is_remaining_list[i]
    removed_feats_columns = np.asarray(df_columns_list[3:])[np.asarray(is_removed_list, dtype=bool)]
    for col in removed_feats_columns:
        # print(col)
        # df.drop([col], axis=1, inplace=True)
        del df[col]

    # df_columns_list_arr = np.asarray(df_columns_list)
    # new_df = pd.DataFrame(index=list(range(df_feat_data.shape[0])), columns=list(df_columns_list_arr[:2] + df_columns_list_arr[is_remaining_list]))
    # for col in df_columns_list_arr[is_remaining_list]:
    #     new_df[col] = df[col]
    df.to_csv(filtered_feats_save_path, index=False)


def filter_features_with_cv(rectified_save_path, filtered_feats_save_path):
    """读取特征文件"""
    df = pd.read_csv(rectified_save_path)
    df_columns_list = df.columns.tolist()
    df_feat_data = np.asarray(df.values[:, 3:], dtype=np.float64)
    df_label_data = np.asarray(df.values[:, 2])
    is_remaining_list = np.asarray([1] * (len(df_columns_list) - 3))

    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                                                                        dtype=bool)])


    """使用ANOVA方法来筛选一部分特征"""
    is_remaining_list = anova_filtered_features(df_feat_data, df_label_data,
                                                             is_remaining_list)
    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                             dtype=bool)])

    """使用皮尔森系数来过滤特征"""
    is_remaining_list = pearson_filtered_features(df_feat_data, is_remaining_list)
    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                                                                        dtype=bool)])

    #
    """使用基于最小二乘法的后向特征筛选"""
    is_remaining_list = ols_filtered_features(df_feat_data, df_label_data, df_columns_list[3:], is_remaining_list)
    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                                                                        dtype=bool)])

    """使用XGBoost来进行递归特征筛选"""
    is_remaining_list = xgboost_rfecv_filtered_features(df_feat_data, df_label_data, df_columns_list[3:],
                                                      is_remaining_list)
    count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
                                                                        dtype=bool)])

    """保存最终过滤后的特征文件"""
    is_removed_list = [0] * len(is_remaining_list)
    for i in range(len(is_remaining_list)):
        is_removed_list[i] = 1 - is_remaining_list[i]
    removed_feats_columns = np.asarray(df_columns_list[3:])[np.asarray(is_removed_list, dtype=bool)]
    for col in removed_feats_columns:
        print(col)
        # df.drop([col], axis=1, inplace=True)
        del df[col]

    df.to_csv(filtered_feats_save_path, index=False)


def test_repeated_features(rectified_unrepeated_save_path='./merged_features/z_score_filtered_features_rectified_unrepeated.csv'):
    """读取特征文件"""

    df = pd.read_csv(rectified_unrepeated_save_path)
    print("sss", df.shape)
    df_columns_list = df.columns.tolist()
    df_feat_data = np.asarray(df.values[:, 3:], dtype=np.float64)
    df_label_data = np.asarray(df.values[:, 2])

    X = df_feat_data
    Y = df_label_data
    # repeated
    for i in range(df_feat_data.shape[-1]):
        for j in range(i + 1, df_feat_data.shape[-1]):
            if np.all(X[:, i] == X[:, j]):
                print("Shot!", i, " ", df_columns_list[i+3], ":",
                      j, " ", df_columns_list[j+3])


if __name__ == '__main__':

    original_csv_file = global_var.base_sklearn_prefix + '/features_saved/merged_features/merged_features.csv'
    
    # 进行归一化过滤（过滤了部分特征，并对过滤后的剩余特征进行了归一化）
    z_score_filtered_path =  global_var.base_sklearn_prefix + '/features_saved/merged_features/z_score_filtered_features.csv'
    # filter_features_by_z_score(original_csv_file, z_score_filtered_path)

    # sys.exit(0)

    # # 修正特征数据中的异常值和缺失值
    rectified_save_path = global_var.base_sklearn_prefix + '/features_saved/merged_features/rectified_features.csv'
    # rectify_missing_and_abnormal_value(z_score_filtered_path, rectified_save_path)

    # sys.exit(0)

    # # 移除重复值
    rectified_unrepeated_save_path = global_var.base_sklearn_prefix + '/features_saved/merged_features/rectified_unrepeated.csv'
    # remove_repeated_and_constant_features(rectified_save_path, rectified_unrepeated_save_path=rectified_unrepeated_save_path)
    # test_repeated_features(rectified_unrepeated_save_path)

    # sys.exit(0)

    # n_features_to_select = 25
    # filter_features(rectified_unrepeated_save_path, global_var.base_sklearn_prefix + '/features_saved/merged_features/filtered_features_' + \
    #         str(n_features_to_select) + '.csv', n_features_to_select = n_features_to_select)
    # sys.exit()

    # n_features_to_select = 10
    # filter_features(rectified_unrepeated_save_path, global_var.base_sklearn_prefix + '/features_saved/merged_features/filtered_features_' + \
    #         str(n_features_to_select) + '.csv', n_features_to_select = n_features_to_select)
    # sys.exit()

    n_features_to_select = 20
    filter_features(rectified_unrepeated_save_path, global_var.base_sklearn_prefix + '/features_saved/merged_features/filtered_features_' + \
            str(n_features_to_select) + '.csv', n_features_to_select = n_features_to_select)

    n_features_to_select = 30
    filter_features(rectified_unrepeated_save_path, global_var.base_sklearn_prefix + '/features_saved/merged_features/filtered_features_' + \
            str(n_features_to_select) + '.csv', n_features_to_select = n_features_to_select)

    filter_features_with_cv(rectified_save_path, global_var.base_sklearn_prefix + '/features_saved/merged_features/filtered_features_with_cv.csv')
   





