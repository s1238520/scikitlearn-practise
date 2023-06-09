from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris=datasets.load_digits()
print(iris.data.shape)
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0)


clf=KNeighborsClassifier(n_neighbors=3,p=2,weights='distance',algorithm='brute')
clf.fit(X_train,y_train)
print("實際:",y_test)
print("預測結果",clf.predict(X_test))
print("準確率:",clf.score(X_test,y_test))

accuracy = []

print(len(X_train))
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train)                 
    y_pred = knn.predict(X_test)              
    accuracy.append(metrics.accuracy_score(y_test, y_pred)) 
    print("n_neighbor:",k,"準確率:",metrics.accuracy_score(y_test, y_pred))

k_range = range(1,100)
plt.plot(k_range, accuracy)
plt.show()