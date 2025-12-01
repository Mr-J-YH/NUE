#!/usr/bin/env python
# coding: utf-8

# In[20]:
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_data = pd.read_excel('训练集.xlsx', index_col=0)

# In[21]:
# 查看灾难类型比例
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(4,4),dpi=100)
plt.pie(train_data.value_counts('标签'),
        explode=(0.01, 0.04, 0.04),
        labels=train_data.value_counts('标签').index,
        autopct='%1.1f%%',
        startangle=90)
plt.title('标签占比', fontsize=12, fontweight='bold')
plt.ylabel('', fontsize=12, fontweight='bold')
plt.show()

# In[22]:
# 数据集拆分
X, y = train_data.drop('标签', axis=1), train_data['标签']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# In[23]:
# 采用默认参数建模
from sklearn import naive_bayes
# 模型初始化
gnb = naive_bayes.GaussianNB()
# 模型拟合
gnb.fit(X_train, y_train)
# 模型在测试数据集上的预测
gnb_pred = gnb.predict(X_test)
# 混淆矩阵
plt.figure(figsize=(5, 4), dpi=100)
sns.heatmap(pd.DataFrame(confusion_matrix(y_test, gnb_pred)), fmt='g',annot=True, cmap='Blues')
plt.title('朴素贝叶斯', fontsize=12)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# 测试集模型评估
print(classification_report(y_test, gnb_pred))
print('精准值：', metrics.precision_score(y_test, gnb_pred, average='weighted'))
print('F1:', metrics.f1_score(y_test, gnb_pred, average='weighted'))
kappa = cohen_kappa_score(y_test,gnb_pred)
print('Kappa: %s' % kappa)
ham_distance=hamming_loss(y_test, gnb_pred)
print('ham_distance: %s' % ham_distance)

# 结果保存（仅保留训练集+测试集结果，删除预测集保存）
# 训练集结果输出
X_train['训练结果'] = y_train
X_train['label'] = y_train
X_train['flag'] = 'train'
X_test['训练结果'] = gnb_pred
X_test['label'] = y_test
X_test['flag'] = 'test'
train_res = pd.concat([X_train, X_test], axis=0)
train_res.to_excel('训练结果_NB.xlsx')