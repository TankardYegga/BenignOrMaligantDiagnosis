# -*- encoding: utf-8 -*-
"""
@File    : extra.py
@Time    : 11/13/2021 1:22 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/mnt/520/lijingyu/lijingyu/zlw/BenignOrMaligantDiagnosis/')
from global_var import *


def z_score_by_scaler(csv_file):
    df = pd.read_csv(csv_file)
    df_cols = df.columns.tolist()
    for col in df_cols[2:]:
        df[[col]] = StandardScaler().fit_transform(df[[col]])
    df.to_csv(save_path, index=True)


original_csv_file = base_feature_prefix + '/merged_features/merged_features.csv'
save_path = base_feature_prefix + '/merged_features/z_scored_merged_features.csv'
z_score_by_scaler(original_csv_file)