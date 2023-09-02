# -*- encoding: utf-8 -*-
"""
@File    : feature_selection.py
@Time    : 11/3/2021 10:20 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import math
import sys
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
                print('topo feats:', feat)
                is_topo = True
                topo_feats.append(feat)
                break
        if is_topo:
            continue

        unrecognized_feats.append(feat)

    print("shape_feats:", len(shape_feats))
    print("texture_feats:", len(texture_feats))
    print("topo_feats:", len(topo_feats))
    print("unrecognized_feats", len(unrecognized_feats))
    return shape_feats, texture_feats, topo_feats, unrecognized_feats


def z_score_filter(csv_file, feat_filtering_threshold = 20, z=3, eps=1e-7):
    features = pd.read_csv(csv_file)
    feat_values = features.values
    print("feat_values type:", feat_values.dtype)

    """
    打印原始特征信息
    """
    original_feats = list(features.columns)
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

    mean_feat_data = np.mean(feat_data, axis=0)
    mean_feat_data = mean_feat_data[np.newaxis, :]
    mean_feat_data = np.tile(mean_feat_data, (feat_data.shape[0], 1))

    std_feat_data = np.std(feat_data, axis=0)
    std_feat_data = std_feat_data[np.newaxis, :]
    std_feat_data = np.tile(std_feat_data, (feat_data.shape[0], 1))

    # z_scored_feat_data = np.subtract(feat_data, mean_feat_data) / std_feat_data
    z_scored_feat_data = (feat_data - mean_feat_data) / ( std_feat_data + eps )
    z_scored_feat_data_df = pd.DataFrame(z_scored_feat_data)
    z_scored_feat_data_df.to_csv('./merged_features/z_score_diy.csv', index=True)

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

    print(to_be_removed_feats)
    print(len(to_be_removed_feats))

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

    # 进行标准化
    if is_standardized:
        standard_normalization_scaler = lambda x: (x - np.mean(x)) / np.std(x)
        # standard_normalization_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
        to_be_scaled_columns = filtered_feats_df.columns.tolist()
        to_be_scaled_columns.remove('image')
        to_be_scaled_columns.remove('label')
        for to_be_scaled_column in to_be_scaled_columns:
            # x = filtered_feats_df[to_be_scaled_column]
            # filtered_feats_df[to_be_scaled_column] = (x - np.mean(x)) / np.std(x)
            filtered_feats_df[[to_be_scaled_column]] = filtered_feats_df[[to_be_scaled_column]].apply(standard_normalization_scaler)
        # 进行缺失值和异常值的处理

    # 保存文件
    filtered_feats_df.to_csv(save_csv_path, index=True)


def z_score_by_df_apply(csv_file):
    df = pd.read_csv(csv_file)
    standard_normalization_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    df_columns = df.columns.tolist()
    df_columns.remove('image')
    df_columns.remove('label')
    df[df_columns] = df[df_columns].apply(standard_normalization_scaler)
    df.to_csv('./merged_features/z_score_apply.csv', index=True)


def z_score_by_standard_scaler(csv_file):
    df = pd.read_csv(csv_file)
    feat_mat = df.values[:, 2:]
    feat_mat = np.asarray(feat_mat, dtype=np.float64)
    feat_mat = StandardScaler().fit_transform(feat_mat)
    print("type :", type(feat_mat))
    z_scored_feat_arr = pd.DataFrame(feat_mat)
    z_scored_feat_arr.to_csv('./merged_features/z_score_standard_scaler.csv', index=True)


def rectify_missing_and_abnormal_value(csv_file, rectified_save_path, z = 3):
    df = pd.read_csv(csv_file)
    df_columns_list = df.columns.tolist()
    """因为df.to_csv时会默认加上一列表示行号，所以读取使用这种方法保存的文件时要注意第一列是否会干扰操作"""
    df.drop(axis=1, columns=df_columns_list[0], inplace=True)
    df_columns_list.remove('image')
    df_columns_list.remove('label')
    print("dir df:", dir(df))
    # 因为多增加了行号列，所以需要从3而不是2开始索引
    feat_mat = np.asarray(df.values[:, 3:], dtype=np.float64)

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
        df[df_columns_list[col_idx + 2]] = single_feat_rectified

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
    plt.show()
    plt.savefig("anova_filter.jpg", dpi=300, pad_inches=0)

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
    plt.savefig('pearson_filter.jpg')
    # plt.show()

    """依据矩阵来筛选特征"""
    # 这里面会动态的改变数组的大小 那么遍历的时候肯定会出问题啊
    re_linear_feats_idxes = np.array([], dtype=np.uint8)
    pearson_filtered_feats_idxes = []
    for i in range(num_of_feats - 1, -1, -1):
        pearson_filtered_feats_idxes.append(i)
        if  i == 0:
            continue
        if i not in re_linear_feats_idxes:
            max_linear = - sys.maxsize - 1
            max_relinear_idx = -1
            for j in range(0, i):
                if j != i and j not in re_linear_feats_idxes and pearson_arr[i, j] > max_linear:
                    max_linear = pearson_arr[i, j]
                    max_relinear_idx = j
            if max_relinear_idx != -1:
                re_linear_feats_idxes = np.append(re_linear_feats_idxes, max_relinear_idx)

    for idx in range(len(re_linear_feats_idxes)):
        try:
            pearson_filtered_feats_idxes.remove(re_linear_feats_idxes[idx])
            if len(pearson_filtered_feats_idxes) != len(np.unique(pearson_filtered_feats_idxes)):
                print('error', len(pearson_filtered_feats_idxes), ":", len(np.unique(pearson_filtered_feats_idxes)))
        except Exception as e:
            print('feat idx:', re_linear_feats_idxes[idx])
            print('idx:', idx)
            print(e)
    for i in range(len(pearson_filtered_feats_idxes)):
        for j in range(i+1, len(pearson_filtered_feats_idxes)):
            if pearson_filtered_feats_idxes[i] == pearson_filtered_feats_idxes[j]:
                print('re:', i, ":", j)
    print('len1:', len(pearson_filtered_feats_idxes))
    pearson_filtered_feats_idxes_cp = np.asarray(pearson_filtered_feats_idxes, dtype=np.int32)
    print("type:", pearson_filtered_feats_idxes_cp.dtype)
    print( pearson_filtered_feats_idxes == pearson_filtered_feats_idxes_cp )
    print( (pearson_filtered_feats_idxes == pearson_filtered_feats_idxes_cp).nonzero()[0] )
    print( len((pearson_filtered_feats_idxes == pearson_filtered_feats_idxes_cp).nonzero()[0]) )
    for i in range(len(pearson_filtered_feats_idxes)):
        print("original:", pearson_filtered_feats_idxes[i])
        print("changed:", pearson_filtered_feats_idxes_cp[i])

    # print('len2:', len(pearson_filtered_feats_idxes_cp))
    # for i in range(len(pearson_filtered_feats_idxes_cp)):
    #     for j in range(i + 1, len(pearson_filtered_feats_idxes_cp)):
    #         if pearson_filtered_feats_idxes_cp[i] == pearson_filtered_feats_idxes_cp[j]:
    #             print('re2:', i, " ", pearson_filtered_feats_idxes_cp[i], ":", " ",
    #                   pearson_filtered_feats_idxes_cp[j], j)
    #             print(pearson_filtered_feats_idxes[i], ":", pearson_filtered_feats_idxes[j])

    print('len4:', len(valid_feats_indexes))
    print('len5', len(np.unique(valid_feats_indexes)))
    remaining_feats_indexes = valid_feats_indexes[pearson_filtered_feats_idxes]
    print('len3', len(remaining_feats_indexes))
    for i in range(len(remaining_feats_indexes)):
        for j in range(i + 1, len(remaining_feats_indexes)):
            if remaining_feats_indexes[i] == remaining_feats_indexes[j]:
                print('re2:', i, ":", j)

    is_remaining_list = np.asarray([1 if i in remaining_feats_indexes else 0 for i in range(len(is_remaining_list))])
    print("len of remaining feats after pearson filtering:", len(is_remaining_list.nonzero()[0]))

    return is_remaining_list


def ols_filtered_features(df_feat_data, df_label_data, df_columns, is_remaining_list, step = 1):
    valid_feats_indexes = []
    for i in range(len(is_remaining_list)):
        if is_remaining_list[i]:
            valid_feats_indexes.append(i)
    print("len of remaining feats before pearson filtering:", len(is_remaining_list.nonzero()[0]))

    valid_feats_indexes = np.asarray(valid_feats_indexes)
    df_label_data = np.asarray([0 if label == 'Benign' else 1 for label in df_label_data])
    df_feat_data = df_feat_data[:, valid_feats_indexes]

    X = df_feat_data
    Y = df_label_data
    # repeated
    repeated_idx = set()
    repeated_pairs = []
    for i in range(1, 436):
        for j in range(i + 1, 436):
            if np.all(X[:, i] == X[:, j]):
                # print("Shot!", valid_feats_indexes[i], " ", df_columns[valid_feats_indexes[i]], ":",
                #       valid_feats_indexes[j], " ", df_columns[valid_feats_indexes[j]])
                print("Shot!", i, " ", df_columns[valid_feats_indexes[i]], ":",
                      j, " ", df_columns[valid_feats_indexes[j]])
                repeated_idx.add(i)
                repeated_idx.add(j)
                if i < j:
                    repeated_pairs.append((i, j))
                else:
                    repeated_pairs.append((j, i))
    print(repeated_idx)
    print(len(repeated_idx))
    for repeated_pair in repeated_pairs:
        i = repeated_pair[0]
        j = repeated_pair[1]
        if i in repeated_idx and j in repeated_idx:
            repeated_idx.remove(j)
        elif i in repeated_idx:
            continue
        elif j in repeated_idx:
            continue
        else:
            continue
    print(repeated_idx)
    #
    # used_idx = list(range(df_feat_data.shape[-1]))
    # for idx in repeated_idx:
    #     used_idx.remove(idx)
    # used_idx = np.asarray(used_idx)

    # T = np.concatenate((X[:, :233], X[:, 234:241]), axis=1)
    # T = np.concatenate((T, X[:, 244:]), axis=1)
    # est = sm.OLS(Y, sm.add_constant(T)).fit()
    # print("est:", est.summary())
    # print("est params:", est.params)

    # removed_i = []
    # # for i in range(df_feat_data.shape[-1]):
    # # for i in range(435, df_feat_data.shape[-1]):
    # # for i in range(435, df_feat_data.shape[-1]):
    # print("len:", df_feat_data.shape[-1])
    # for i in range(1, df_feat_data.shape[-1]):
    #     # est = sm.OLS(Y, sm.add_constant( np.concatenate((X[:, :434], X[:, 435:]), axis=1) ) ).fit()
    #     est = sm.OLS(Y, sm.add_constant(X[:, 3:i])).fit()
    #      # print("est:", est.summary())
    #     # print("est params:", est.params)
    #     print('pvalues:', est.pvalues[:10])
    #     if True in np.isnan(est.pvalues):
    #         removed_i.append(i)
    # print('removed i ', removed_i)

    # est = sm.OLS(Y, sm.add_constant(np.concatenate( (X[:, 434:435],X[:, 434:435]), axis=1)) ).fit()
    # est = sm.OLS(Y, sm.add_constant(X[:, 435:872])).fit()
    # est = sm.OLS(Y, sm.add_constant(X[:, 872:])).fit()
    # est = sm.OLS(Y, sm.add_constant(np.concatenate((X[:, :435], X[:, 872:]), axis=1))).fit()

    # removed_i = []
    # for i in range(df_feat_data.shape[-1]):
    #     est = sm.OLS(Y, sm.add_constant(X[:, used_idx], has_constant='skip'), missing='raise').fit()
    #     # print("type est:", type(est))
    #     # print("est:", est.summary())
    #     # print("t values:", est.tvalues)
    #     if True in np.isnan(est.pvalues):
    #         removed_i.append(i)
    # print("removed i:", removed_i)


    # removed_i = []
    # for i in range(df_feat_data.shape[-1]):
    #     # if i - 1 >= 0 and i + 1 < df_feat_data.shape[-1]:
    #     #     x_left = X[:, :i]
    #     #     x_right = X[:, i+1:]
    #     #     new_x = np.concatenate( (x_left, x_right), axis=1)
    #     # elif i - 1 >= 0:
    #     #     new_x = X[:, :i]
    #     # elif i + 1 < df_feat_data.shape[-1]:
    #     #     new_x = X[:, i + 1:]
    #     # else:
    #     #     new_x = X
    #     new_x = X[:, i:i+1]
    #     print('new X shape:', new_x.shape)
    #     est = sm.OLS(Y, sm.add_constant(new_x, has_constant='skip'), missing='raise').fit()
    #     # print("type est:", type(est))
    #     # print("est:", est.summary())
    #     print("I:", i)
    #     print("tvalues:", est.pvalues[:10])
    #     print("bse:", est.bse[:10])
    #     if True in np.isnan(est.bse):
    #         print("i:", i, " has nan")
    #         removed_i.append(i)
    #     # print("t values:", est.tvalues)
    #     # if False in np.isnan(est.pvalues):
    #     #     removed_i.append(i)
    # print("removed i:", removed_i)

        # print("p values:", est.pvalues)
    # print("est params:", est.params)  # 434
    # print("est t values:", est.tvalues)
    # print("est fvalue:", est.fvalue)
    # print("est fpvalue:", est.f_pvalue)
    # print("est pvalue:", est.pvalues)
    # print("est t_tests:", est.t_test([1, 1]))
    # print("est f_tests:", est.f_test(np.identity(2)))
    # print(df_columns[valid_feats_indexes[435]])
    # print(df_columns[valid_feats_indexes[871]])
    # for i in range(435, 872):
    #     for j in range(i+1, 872):
    #         if np.all(X[:, i] == X[:, j]):
    #             print("Shot!", i, ":", j)

    # remaining_feats_original_idxex = valid_feats_indexes
    #
    # est = sm.OLS(Y, sm.add_constant(X[:, remaining_feats_original_idxex])).fit()
    # feat_pvalues_arr = est.pvalues
    # inapparent_feats_num = len(( feat_pvalues_arr >= 0.05).nonzero()[0])
    # inapparent_feats_dict = dict(zip(np.where(feat_pvalues_arr >= 0.05)[0].tolist(),
    #                                 feat_pvalues_arr[feat_pvalues_arr >= 0.05].tolist()))
    # while inapparent_feats_num:
    #     inapparent_feats_idxes = []
    #     inapparent_feats_dict = sorted(inapparent_feats_dict, lambda item: item[1], reverse=True)
    #     removed_feats_num = min(step, inapparent_feats_num)
    #     for i in range(removed_feats_num):
    #         inapparent_feats_idxes.append(list(inapparent_feats_dict.keys())[i])
    #
    #     # 这里的问题是怎么把这些指定idx的位置给删除掉
    #     # 你把一个位置移除了，就会导致整个list的索引空间发生变化
    #     # 那只能做一份拷贝，利用移除值的方式来删除
    #     remaining_feats_original_idxex_cp = list(remaining_feats_original_idxex).copy()
    #     for idx in inapparent_feats_idxes:
    #         remaining_feats_original_idxex_cp.remove(remaining_feats_original_idxex[idx])
    #     remaining_feats_original_idxex = np.asarray(remaining_feats_original_idxex_cp)
    #
    #     # 将这些特征移除掉再次计算est
    #     est = sm.OLS(Y, sm.add_constant(X[:, remaining_feats_original_idxex])).fit()
    #     feat_pvalues_arr = est.pvalues
    #     inapparent_feats_num = len((feat_pvalues_arr >= 0.05).nonzero()[0])
    #     inapparent_feats_dict = dict(zip(np.where(feat_pvalues_arr >= 0.05)[0].tolist(),
    #                                      feat_pvalues_arr[feat_pvalues_arr >= 0.05].tolist()))
    #
    #
    #
    # is_remaining_list = np.asarray([1 if i in remaining_feats_original_idxex else 0 for i in range(len(is_remaining_list))])
    # print("len of remaining feats after pearson filtering:", len(is_remaining_list.nonzero()[0]))

    return is_remaining_list


def xgboost_rfe_filtered_features(df_feat_data, df_label_data, df_columns, is_remaining_list):
    valid_feats_indexes = []
    for i in range(len(is_remaining_list)):
        if is_remaining_list[i]:
            valid_feats_indexes.append(i)
    print("len of remaining feats before pearson filtering:", len(is_remaining_list.nonzero()[0]))

    valid_feats_indexes = np.asarray(valid_feats_indexes)
    df_feat_data = df_feat_data[:, valid_feats_indexes]
    df_label_data = np.asarray([0 if label == 'Benign' else 1 for label in df_label_data])

    selector = RFE(estimator=XGBClassifier(), n_features_to_select=20, step=1)
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
    print("selected feature_importances::", selector.estimator_.feature_importances_[selector.support_])
    print("model coef:", selector.estimator_.coef_)

    fig, axes = plt.subplots(1, 1, figsize=(20, 20))
    data = pd.Series(selector.estimator.feature_importances_[selector.support_], index=df_columns)
    data.plot.barh(ax=axes[0, 0], color='r', alpha=0.5)
    plt.savefig("xgboost_feats_importances.jpg")
    plt.show()

    remaining_feats_indexes = valid_feats_indexes[selector.support_]
    is_remaining_list = np.asarray([1 if i in remaining_feats_indexes else 0 for i in range(len(is_remaining_list))])
    print("len of remaining feats after pearson filtering:", len(is_remaining_list.nonzero()[0]))
    return is_remaining_list


def xgboost_rfecv_filtered_features(df_feat_data, df_label_data, df_columns, is_remaining_list):
    valid_feats_indexes = []
    for i in range(len(is_remaining_list)):
        if is_remaining_list[i]:
            valid_feats_indexes.append(i)
    print("len of remaining feats before pearson filtering:", len(is_remaining_list.nonzero()[0]))

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
    print("n_features_to_select:", selector.n_features_to_select)
    print("n_features_in_", selector.n_features_in_)
    print("n_features:",  (selector.ranking_ == 1).nonzero()[0])
    print("model feature importances:", selector.estimator_.feature_importances_)  # 单纯使用model会显示所有特征重要性的得分，因为不会像RFE一样对特征进行筛选
    print("selected feature_importances::", selector.estimator_.feature_importances_[selector.support_])
    print("model coef:", model.coef_)

    fig, axes = plt.subplots(1, 1, figsize=(20, 20))
    data = pd.Series(model.feature_importances_[selector.support_], index=df_columns)
    data.plot.barh(ax=axes[0, 0], color='r', alpha=0.5)
    plt.savefig("xgboost_cv_feats_importances.jpg")
    plt.show()

    remaining_feats_indexes = valid_feats_indexes[selector.support_]
    is_remaining_list = np.asarray([1 if i in remaining_feats_indexes else 0 for i in range(len(is_remaining_list))])
    print("len of remaining feats after pearson filtering:", len(is_remaining_list.nonzero()[0]))
    return is_remaining_list


def preprocess_features(original_csv_file):

    """通过Z分法找出需要被删除的特征"""
    to_be_removed_feats = z_score_filter(original_csv_file)

    """根据被删除的特征来保存标准化后的新的特征文件"""
    save_csv_path = './merged_features/z_score_filtered_features.csv'
    save_filtered_features(original_csv_file, save_csv_path, to_be_removed_feats)

    """调用函数库来实现z分法，但是没有特征删除操作，只是为了测试下库函数的使用效果"""
    z_score_by_df_apply(original_csv_file)
    z_score_by_standard_scaler(original_csv_file)

    """修正特征数据中的异常值和缺失值"""
    rectified_save_path = './merged_features/z_score_filtered_features_rectified.csv'
    rectify_missing_and_abnormal_value(save_csv_path, rectified_save_path)
    return rectified_save_path


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


def filter_features(rectified_save_path, filtered_feats_save_path):
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
    # is_remaining_list = ols_filtered_features(df_feat_data, df_label_data, df_columns_list[3:], is_remaining_list)
    # count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
    #                                                                     dtype=bool)])

    """使用XGBoost来进行递归特征筛选"""
    # is_remaining_list = xgboost_rfe_filtered_features(df_feat_data, df_label_data, df_columns_list[3:], is_remaining_list)
    # count_different_feat_num(np.asarray(df_columns_list[3:])[np.asarray(is_remaining_list,
    #                                                                     dtype=bool)])

    """保存最终过滤后的特征文件"""
    # df_columns_list_arr = np.asarray(df_columns_list)
    # new_df = pd.DataFrame(index=list(range(df_feat_data.shape[0])), columns=list(df_columns_list_arr[:2] + df_columns_list_arr[is_remaining_list]))
    # for col in df_columns_list_arr[is_remaining_list]:
    #     new_df[col] = df[col]
    # new_df.to_csv(filtered_feats_save_path, index=True)


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

    original_csv_file = './merged_features/merged_features.csv'

    rectified_save_path = './merged_features/z_score_filtered_features_rectified.csv'
    # rectified_save_path = preprocess_features(original_csv_file)

    rectified_unrepeated_save_path = 'rectified_unrepeated.csv'
    # remove_repeated_and_constant_features(rectified_save_path, rectified_unrepeated_save_path=rectified_unrepeated_save_path)
    # test_repeated_features(rectified_unrepeated_save_path)

    filter_features(rectified_unrepeated_save_path, '')





