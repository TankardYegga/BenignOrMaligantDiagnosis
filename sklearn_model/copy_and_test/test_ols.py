# -*- encoding: utf-8 -*-
"""
@File    : test_ols.py
@Time    : 11/8/2021 2:51 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm #方法一
import statsmodels.formula.api as smf  # 方法二
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv(r'D:\AllExploreDownloads\IDM\ExtraCode\merged_features\test_ols.csv', index_col=0)
X = df[['TV', 'test1','test2']]
y = df['sales']

est = sm.OLS(y, sm.add_constant(X)).fit() #方法一
# est = smf.ols(formula='sales ~ TV + test1', data=df).fit()  # 方法二
y_pred = est.predict(X)

df['sales_pred'] = y_pred
print(df)
print(est.summary())  # 回归结果
print(est.params)  # 系数

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')  # ax = Axes3D(fig)
# ax.scatter(X['TV'], X['radio'], y, c='b', marker='o')
# ax.scatter(X['TV'], X['radio'], y_pred, c='r', marker='+')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()