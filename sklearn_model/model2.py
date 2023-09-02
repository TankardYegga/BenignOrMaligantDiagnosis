# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 11/3/2021 10:21 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import sys
from scipy.optimize.linesearch import LineSearchWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from __init__ import global_var

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', LineSearchWarning)


# 在整个数据集上进行交叉验证得到k折的平均分
# 并在整个训练集上进行测试
def cv_train_model_on_whole(model, param_grid, x, y, scoring='roc_auc', n_fold=10, model_name='unamed_model'):
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1)
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        n_jobs=-1,
        # cv=n_fold,
        cv=kfold
    )
    grid_result = grid_search.fit(x, y)
    with open(global_var.base_sklearn_prefix + "/model_saved/" + model_name + '.pkl', 'wb') as f:
        pickle.dump(grid_result, f)
        print(model_name + '模型保存成功')

    print("*" * 100)
    print("model used:", model_name)
    print("number of features:", x.shape[-1])
    print("Best Scores:", grid_search.best_score_)
    print("Best Params:", grid_search.best_params_)

    # print("x is", x[4])
    # x = x[4].reshape(1, x[4].shape[0])
    # y = y[4].reshape(1, x[4].shape[0])

    # sys.exit(0)

    # y_predict_prob = grid_search.predict_proba(x)
    y_predict = grid_search.predict(x)
    print("x:", x)
    print('predict:', y_predict)
    print("y:", y)
    # for i in range(len(y)):
    #     print('cmp: ', y[i], ':', y_predict[i])
    print('equal', np.all(y_predict == y))
    print("acc:", accuracy_score(y, y_predict, normalize=True))
    print("auc:", roc_auc_score(y, y_predict))

    # with open(global_var.base_sklearn_prefix + "/model_saved/" + model_name + '.pkl', 'rb') as f:
    #     net = pickle.load(f)
    #     y_predict = net.predict(x)

    #     print('equal', np.all(y_predict == y))
    #     print("acc:", accuracy_score(y, y_predict, normalize=True))
    #     print("auc:", roc_auc_score(y, y_predict))
    
    #     sys.exit(0)

    # precision_recall_curve(y, y_predict)
    # roc_curve(y, y_predict)
    # matthews_corrcoef(y, y_predict)
    # print('y_predict', y_predict)
    # means = grid_search.cv_results_['mean_test_score']
    # params = grid_search.cv_results_['params']
    # for mean_score, param in zip(means, params):
    #     print(f'with {param}: {mean_score}')
    print("*" * 100)


# 将整个数据集划分为训练集和测试集
# 在训练集上进行交叉验证的训练
# 然后再测试集上进行测试
def cv_train_model_on_part(model, param_grid, train_x, train_y, test_x, test_y, scoring='roc_auc', n_fold=10, model_name='unamed_model'):
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1)
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        n_jobs=-1,
        # cv=n_fold,
        cv=kfold,
    )
    grid_result = grid_search.fit(train_x, train_y)
    with open(global_var.base_sklearn_prefix + "/model_saved/" + model_name + '.pkl', 'wb') as f:
        pickle.dump(grid_result, f)
        print(model_name + '模型保存成功')

    print("*" * 100)
    print("model used:", model_name)
    print("number of features:", train_x.shape[-1])
    print("Best Scores:", grid_search.best_score_)
    print("Best Params:", grid_search.best_params_)

    test_y_predict = grid_search.predict(test_x)
    print('y predict result:', test_y_predict)
    print('equal', np.all(test_y==test_y_predict))
    print("acc:", accuracy_score(test_y, test_y_predict, normalize=True))
    print("auc:", roc_auc_score(test_y,  test_y_predict))
    
    print("*" * 100)


def obtain_data(feats_csv_file):
    df = pd.read_csv(feats_csv_file)
    x = df.values[:, 3:]
    y = df.values[:, 2]
    imgs = df.values[:, 1]
    print("imgs:", imgs)

    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    label_encoder_y = label_encoder.transform(y)
    return x, label_encoder_y


def obtain_data_by_specified_feats(csv_file, type='whole'):
    """
    :param csv_file:
    :param type: { whole, texture, topo}
    :return:
    """
    texture_feats = ['wavelet-HH_ngtdm_Busyness', 'wavelet-HH_glcm_Imc2',
                     'wavelet-LH_firstorder_Maximum', 'lbp-2D_glrlm_RunEntropy',
                     'log-sigma-3-mm-3D_firstorder_Uniformity', 'log-sigma-3-mm-3D_firstorder_Energy',
                     'log-sigma-2-mm-3D_glszm_ZoneEntropy', 'log-sigma-1-mm-3D_firstorder_Mean',
                     'original_gldm_DependenceVariance', 'original_glcm_ldmn'
                     ]
    topo_feats = ['AverageClusteringCoefficient_53', 'PercentageofIsolatedPoints_41',
                  'MaximumVertexDegree_41', 'Diameter_36',
                  'AverageVertexEccentricity_32',
                  'AverageVertexEccentricity_27',
                  'GiantConnectedComponentRatio_26',
                  'AverageClusteringCoefficient_16',
                  'AverageClusteringCoefficient_12',
                  'MaximumVertexDegree_9']
    merged_feats = texture_feats + topo_feats
    df = pd.read_csv(csv_file)
    df_cols = df.columns.tolist()
    df_feat_data = df.values[:, 3:]
    df_label_data = df.values[:, 2]
    label_encoder = LabelEncoder().fit(df_label_data)
    df_label_data = label_encoder.transform(df_label_data)
    texture_feats_idx = np.asarray([df_cols[3:].index(feat) if feat in df_cols else None for feat in texture_feats])
    topo_feats_idx = np.asarray([df_cols[3:].index(feat) if feat in df_cols else None for feat in topo_feats])
    texture_feats_idx = np.asarray(texture_feats_idx[texture_feats_idx != None], dtype=np.int32)
    topo_feats_idx = np.asarray(topo_feats_idx[topo_feats_idx != None], dtype=np.int32)
    merged_feats_idx = np.append(texture_feats_idx, topo_feats_idx)

    if type == 'texture':
        return df_feat_data[:, texture_feats_idx], df_label_data
    elif type == 'topo':
        return df_feat_data[:, topo_feats_idx], df_label_data
    else:
        return df_feat_data[:, merged_feats_idx], df_label_data


def obtain_data_by_all_feats(csv_file):
    """
    :param csv_file:
    :param type: { whole, texture, topo}
    :return:
    """
    df = pd.read_csv(csv_file)
    df_cols = df.columns.tolist()
    df_feat_data = df.values[:, 3:]
    df_label_data = df.values[:, 2]
    label_encoder = LabelEncoder().fit(df_label_data)
    df_label_data = label_encoder.transform(df_label_data)

    return df_feat_data, df_label_data


if __name__ == '__main__':
    feats_csv_file_1ist = [global_var.base_feature_prefix + '/merged_features/filtered_features_25.csv',
                           global_var.base_feature_prefix + '/merged_features/filtered_features_with_cv.csv',
                           global_var.base_feature_prefix + '/merged_features/filtered_features_10.csv',
                           global_var.base_feature_prefix + '/merged_features/filtered_features_20.csv',
                           global_var.base_feature_prefix + '/merged_features/filtered_features_30.csv',
                           global_var.base_feature_prefix + '/merged_features/filtered_features_9.csv',
                           ]
    file_idx = 2
    feats_csv_file = feats_csv_file_1ist[file_idx]
    texture_feats_num_1ist = [21, 14, 1, 17, 26, 9]
    topo_feats_num_1ist = [4, 3, 9, 3, 4, 0]
    scoring_list = ['roc_auc', 'accuracy', 'precision']
    texture_feats_num = texture_feats_num_1ist[file_idx]
    topo_feats_num = topo_feats_num_1ist[file_idx]

    class_weight = {0: 0.50, 1: 0.50}
    scoring = scoring_list[0]
    n_fold = 10

    data_x, data_y = obtain_data(feats_csv_file)
    # sys.exit(0)
    # csv_file = r'./merged_features_1/merged_features.csv'
    # csv_file = r'./merged_features_1/z_scored_merged_features.csv'
    # data_x, data_y = obtain_data_by_all_feats(csv_file=csv_file)
    print('data_x shape:', data_x.shape)
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size=0.1,
                                                                            random_state=5, stratify=data_y)
    print(len((data_y_test == 1).nonzero()[0]))
    print(len(data_y_test))
    print(len((data_y_train == 1).nonzero()[0]))
    print(len(data_y_train))

    # sys.exit(0)

    # knc_model = KNeighborsClassifier()
    # n_neighbors = [3, 4, 5, 6, 7]
    # leaf_size = list(range(25, 35 + 1))
    # algorithm = ['ball_tree', 'kd_tree']
    # param_grid = dict(n_neighbors=n_neighbors, leaf_size=leaf_size, algorithm=algorithm)
    # cv_train_model_on_whole(knc_model, param_grid,  data_x, data_y, scoring, n_fold, 'knc')
    # cv_train_model_on_whole(knc_model, param_grid,  data_x[:, :texture_feats_num], data_y, scoring, n_fold, 'knc')
    # cv_train_model_on_whole(knc_model, param_grid,  data_x[:, -topo_feats_num:], data_y, scoring, n_fold, 'knc')

    # lr_model = LogisticRegression(
    #     penalty='l2',
    #     verbose=0,
    #     class_weight=class_weight,
    #     random_state=0,
    #     multi_class='auto',
    #     n_jobs=-1,
    #     dual=False
    # )
    # fit_intercept = [True, False]
    # C = list(np.linspace(0, 1, num=10, endpoint=False)) + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # C.remove(0)
    # tol = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    # param_grid = dict(fit_intercept=fit_intercept, C=C, tol=tol, solver=solver)
    # cv_train_model_on_whole(lr_model, param_grid, data_x, data_y, scoring, n_fold, 'lr')
    # cv_train_model_on_whole(lr_model, param_grid, data_x[:, :texture_feats_num], data_y, scoring, n_fold, 'lr')
    # cv_train_model_on_whole(lr_model, param_grid, data_x[:, -topo_feats_num:], data_y, scoring, n_fold, 'lr')

    svc_model = SVC(
        kernel='rbf',
        class_weight=class_weight,
        probability=True,
        # probability=False,
        # gamma=float(1 / 20),
        random_state=1,
    )
    C = list(np.linspace(0, 1, num=10, endpoint=False)) + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    C.remove(0)
    shrinking = [True, False]
    tol = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    param_grid = dict(C=C, shrinking=shrinking, tol=tol)
    # cv_train_model_on_part(svc_model, param_grid, data_x_train, data_y_train, data_x_test, data_y_test,
    #                  scoring, n_fold, 'svc_on_part_25')
    # cv_train_model_on_whole(svc_model, param_grid, data_x, data_y, scoring, n_fold, 'svc_on_whole_25')
    # cv_train_model_on_whole(svc_model, param_grid, data_x, data_y, scoring, n_fold, 'svc_on_whole_cv')
    # cv_train_model_on_whole(svc_model, param_grid, data_x, data_y, scoring, n_fold, 'svc_on_whole_10')
    # cv_train_model_on_whole(svc_model, param_grid, data_x, data_y, scoring, n_fold, 'svc_on_whole_20')
    # cv_train_model_on_whole(svc_model, param_grid, data_x, data_y, scoring, n_fold, 'svc_on_whole_30')
    cv_train_model_on_whole(svc_model, param_grid, data_x, data_y, scoring, n_fold, 'svc_on_whole_10')

    # svc_model = SVC(
    #     kernel='rbf',
    #     class_weight=class_weight,
    #     probability=True,
    #     gamma=float(1 / texture_feats_num),
    #     random_state=0,
    # )
    # cv_train_model_on_whole(svc_model, param_grid, data_x[:, :texture_feats_num], data_y, scoring, n_fold, 'svc')

    # svc_model = SVC(
    #     kernel='rbf',
    #     class_weight=class_weight,
    #     probability=True,
    #     gamma=float(1 / topo_feats_num),
    #     random_state=0,
    # )
    # cv_train_model_on_whole(svc_model, param_grid, data_x[:, -topo_feats_num:], data_y, scoring, n_fold, 'svc')

    # GaussianNB_model = GaussianNB(
    #     priors=None,
    # )
    # var_smoothing = [1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    # param_grid = dict(var_smoothing=var_smoothing)
    # cv_train_model_on_whole(GaussianNB_model, param_grid, data_x, data_y, scoring, n_fold, 'GaussianNB')
    # cv_train_model_on_whole(GaussianNB_model, param_grid, data_x[:, :texture_feats_num], data_y, scoring, n_fold, 'GaussianNB')
    # cv_train_model_on_whole(GaussianNB_model, param_grid, data_x[:, -topo_feats_num:], data_y, scoring, n_fold, 'GaussianNB')

    sys.exit(0)

    rf_model = RandomForestClassifier(random_state=7)
    # n_estimators = list(range(1, 30, 1))
    # criterion = ["gini", "entropy"]
    # bootstrap = [True, False]
    n_estimators = [28]
    criterion = ["entropy"]
    bootstrap = [True]
    # max_samples = list(np.linspace(0.1, 1, num=10, endpoint=False))
    # min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # param_grid = dict(n_estimators=n_estimators, criterion=criterion,
    #                   bootstrap=bootstrap, max_samples=max_samples,
    #                   min_samples_split=min_samples_split)
    param_grid = dict(n_estimators=n_estimators, criterion=criterion,
                                        bootstrap=bootstrap)
    cv_train_model_on_part(rf_model, param_grid, data_x_train, data_y_train, data_x_test, data_y_test,
                   scoring, n_fold, 'rf_model')
    cv_train_model_on_part(rf_model, param_grid, data_x_train[:, :texture_feats_num], data_y_train,
                     data_x_test[:, :texture_feats_num], data_y_test,
                     scoring, n_fold, 'rf_model')
    cv_train_model_on_part(rf_model, param_grid, data_x_train[:, -topo_feats_num:], data_y_train,
                     data_x_test[:, -topo_feats_num:], data_y_test,
                     scoring, n_fold, 'rf_model')
    # cv_train_model_on_whole(rf_model, param_grid, data_x[:, :texture_feats_num], data_y, scoring, n_fold, 'rf_model')
    # cv_train_model_on_whole(rf_model, param_grid, data_x[:, -topo_feats_num:], data_y, scoring, n_fold, 'rf_model')







