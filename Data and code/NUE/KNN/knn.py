from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


train_data = pd.read_excel('训练集.xlsx', index_col=0)

X, y = train_data.drop('标签', axis=1), train_data['标签']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)




# 采用默认参数建模
from sklearn.metrics import classification_report


# 模型初始化
knn = KNeighborsClassifier(n_neighbors=3)
# 模型拟合
knn.fit(X_train, y_train)
# 模型在测试数据集上的预测
knn_pred = knn.predict(X_test)


# ---------------------------------
# 训练集结果输出
X_train['训练结果'] = y_train
X_train['label'] = y_train
X_train['flag'] = 'train'
X_test['训练结果'] = knn_pred
X_test['label'] = y_test
X_test['flag'] = 'test'
train_res = pd.concat([X_train, X_test], axis=0)
train_res.to_excel('训练结果_knn.xlsx')



# 测试集模型评估
print(classification_report(y_test, knn_pred))


## 采用混淆矩阵（metrics）计算各种评价指标
print('精准值：', metrics.precision_score(y_test, knn_pred, average='weighted'))
print('F1:', metrics.f1_score(y_test, knn_pred, average='weighted'))
kappa = cohen_kappa_score(y_test,knn_pred)
print('Kappa: %s' % kappa)
ham_distance=hamming_loss(y_test, knn_pred)
print('ham_distance: %s' % ham_distance)

## 分类报告
#class_report = metrics.classification_report(y_test, knn_pred_1)
#print(class_report)
#
## 输出混淆矩阵
cm = metrics.confusion_matrix(y_test, knn_pred)
print('--混淆矩阵--')
print(cm)
#
## 创建颜色映射，使用蓝色渐变
cmap = plt.cm.Blues
#
## 显示混淆矩阵
sns.heatmap(pd.DataFrame(cm), annot=True, cmap=cmap, fmt='g')
#plt.title('KNN')
plt.title('K最近邻', fontsize=12, fontfamily="SimHei")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
## 显示图形
plt.show()

# 结果保存
# train_data['训练结果'] = knn_pred_1
# train_data.to_excel('训练结果_knn.xlsx')