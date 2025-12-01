#!/usr/bin/env python
# coding: utf-8

# 完整库导入（明确导入classification_report，解决NameError）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, accuracy_score,
                             classification_report)  # 显式导入classification_report
import sklearn.preprocessing
import sklearn.cluster
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, hamming_loss
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']##中文乱码问题！
plt.rcParams['axes.unicode_minus']=False#横坐标负号显示问题！

'''
关于读取数据，
'''
df_name1 = '训练集.xlsx'
df = pd.read_excel(df_name1)

df.dropna(inplace=True)
df.fillna(0, inplace=True)

print(df.head())

y = df['标签']
x = df.drop(columns=['标签'])

def my_dtree(x,y):##3 决策树（仅去掉newdata参数，其余完全保留）
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=71) ##随机划分数据集
    model = DecisionTreeClassifier(criterion= 'gini', random_state=0)
    model.fit(xtrain, ytrain)
    pre_rf = model.predict(xtest)
    pro_rf = model.predict_proba(xtest)
    socre = model.score(xtest, ytest)  # 准确率
    pr1 = pd.DataFrame(model.predict(x), columns=['预测值'])
    return pre_rf, ytest, pro_rf, model.classes_[0], socre, pr1, model, xtrain, ytrain

# 直接调用函数（无newdata参数）
res = my_dtree(x, y)
ypred = res[0]
ytest = res[1]

print('使用决策树预测的准确率为:', res[4])

# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=1234)
# clf = DecisionTreeClassifier(random_state=0)
clf = res[-3]
xtrain = res[-2]
ytrain = res[-1]
clf.fit(xtrain, ytrain)

# 注释掉决策树可视化代码（解决Graphviz报错）
# import graphviz
# from sklearn import tree
# dot = tree.export_graphviz(clf, out_file=None,
#                            feature_names=x.columns,
# #                            class_names = ['劣质','一般','优质'],
#                            filled=True, rounded=True)
# graph = graphviz.Source(dot.replace("helvetica","FangSong"))
# graph.render("决策树")
# graph

# 绘制混淆矩阵
def mat_show(y1, pr1):
    '''
   y1 : y
    pr1 :  pr1 = pd.DataFrame(model.predict(x),columns=['预测值'])
    '''
    C = confusion_matrix(y1, pr1)
    plt.figure(figsize=(6, 6))  #设置图片大小
    plt.matshow(C, cmap=plt.cm.Blues)
    plt.colorbar()
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='top')
    plt.ylabel('True label')
    plt.xlabel('Predicted Label')
    plt.title('决策树', fontsize=12, fontfamily="SimHei")
    plt.show()

# 采用混淆矩阵（metrics）计算各种评价指标
print('精准值：', metrics.precision_score(ytest, ypred, average='weighted'))
print('F1:', metrics.f1_score(ytest, ypred, average='weighted'))
kappa = cohen_kappa_score(ytest, ypred)
print('Kappa: %s' % kappa)
ham_distance = hamming_loss(ytest, ypred)
print('ham_distance: %s' % ham_distance)

# 直接使用导入的classification_report（无需前缀）
print(classification_report(res[0], res[1]))
mat_show(res[0], res[1])