# SVR与SVC的AUC/ROC计算
import numpy as np
from sklearn.svm import SVR,SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import sys

# y_scores=np.array([ 0.63, 0.53, 0.36, 0.02, 0.70 ,1 , 0.48, 0.46, 0.57])
# y_true=np.array(['0', '1', '0', '0', '1', '1', '1', '1', '1'])
# roc_auc_score(y_true, y_scores)

# sys.exit()


iris = datasets.load_iris()
X = iris.data
Y = iris.target
X = X[:100]
Y = Y[:100]

x_train, x_test, y_train, y_test = train_test_split(X, Y,  train_size=0.7)
 
print("------------------------------ SVC ------------------------------------------")
clf = SVC(kernel='rbf', C=100, gamma=0.0001, probability=True)
clf.fit(x_train, y_train)
 
y_train_pre = clf.predict(x_train)
y_test_pre = clf.predict(x_test)
print("Accuracy: "+str(clf.score(x_train,y_train)))  
 
y_train_predict_proba = clf.predict_proba(x_train) #每一类的概率


test_auc = metrics.roc_auc_score(y_train, np.argmax(y_train_predict_proba, 1))
print("test acc:", test_auc)

fpr, tpr, thresholds = metrics.roc_curve(y_train, np.argmax(y_train_predict_proba, 1))
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % test_auc)
plt.savefig("./yes.jpg")
plt.show()

print("test acc2:", auc(fpr,tpr))

sys.exit()



false_positive_rate, recall, thresholds = roc_curve(y_train, y_train_predict_proba[:, 1])

train_auc=auc(false_positive_rate,recall)
print("train AUC: "+str(train_auc))
 
print("------------------------------------")
print("Accuracy: "+str(clf.score(x_test,y_test)))
 
y_test_predict_proba = clf.predict_proba(x_test) #每一类的概率
false_positive_rate, recall, thresholds = roc_curve(y_test, y_test_predict_proba[:, 1])
test_auc=auc(false_positive_rate,recall)
print("test AUC: "+str(test_auc))
 


plt.figure(0)
plt.title('ROC of SVM in test data')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % test_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig('yes2.jpg')
plt.show()
 

print("--------------------------- SVR ------------------------------------------")
 
reg = SVR(kernel='rbf', C=100, gamma=0.0001)
reg.fit(x_train, y_train)
y_train_pre = reg.predict(x_train)
y_test_pre = reg.predict(x_test)
train_auc = metrics.roc_auc_score(y_train,y_train_pre)
print("train AUC: "+str(train_auc))
 
print("--------------------------------")
 
test_auc = metrics.roc_auc_score(y_test,y_test_pre)
print("test AUC: "+str(test_auc))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_test_pre)
 
plt.figure(1)
plt.title('ROC of SVR in test data')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % test_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.savefig("yes3.jpg")
plt.show()