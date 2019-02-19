# coding : utf-8

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from kNNPractice import KNNClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=827)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

y_predict = knn_clf.predict(X_test)
print('准确率：' + str(accuracy_score(y_test, y_predict)))
print('准确率：' + str(knn_clf.score(X_test, y_test)))

print('-------使用自己的kNN----------')

knn_clf2 = KNNClassifier(k=3)
knn_clf2.fit(X_train, y_train)
y_predict2 = knn_clf2.predict(X_test)
print('准确率：' + str(knn_clf2.accuracy_score(y_test, y_predict2)))
print('准确率：' + str(knn_clf2.score(X_test, y_test)))
