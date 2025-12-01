#导入所需数据库
from sklearn import svm
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
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

#读取数据（仅保留训练集，删除预测集读取）
train_data = pd.read_excel('训练集.xlsx', sheet_name='分类')

# 将数据集拆分为训练集和测试集
y = train_data.iloc[:, 1]
X = train_data.iloc[:, 2:]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 12345)

#初始化分类器
clf = svm.SVC(kernel='linear', C=1)

#拟合模型
clf.fit(X_train, y_train)

#预测测试集
y_pred = clf.predict(X_test)

#计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#将预测结果和准确率存储在表格中
res = pd.DataFrame({'预测结果': y_pred, '准确率': [accuracy] * len(y_pred)})

#将表格输出为Excel文件
#res.to_excel('预测结果.xlsx', index=False)

# 采用混淆矩阵（metrics）计算各种评价指标
print('精准值：', metrics.precision_score(y_test, y_pred, average='weighted'))
print('F1:', metrics.f1_score(y_test, y_pred, average='weighted'))
kappa = cohen_kappa_score(y_test,y_pred)
print('Kappa: %s' % kappa)
ham_distance=hamming_loss(y_test, y_pred)
print('ham_distance: %s' % ham_distance)

# 分类报告
class_report = metrics.classification_report(y_test, y_pred)
print(class_report)

# 输出混淆矩阵
#ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print('--混淆矩阵--')
print(cm)
#print(cm)

# 创建颜色映射，使用蓝色渐变
cmap = plt.cm.Blues

# 显示混淆矩阵
sns.heatmap(pd.DataFrame(cm), annot=True, cmap=cmap, fmt='g')
#plt.title('SVM')
plt.title('支持向量机', fontsize=12, fontfamily="SimHei")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# 显示图形
plt.show()