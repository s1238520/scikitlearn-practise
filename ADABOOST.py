from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris=datasets.load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import LinearSVC

#clf=DecisionTreeClassifier()
#clf=LinearSVC()
clf=GaussianNB()
adb=AdaBoostClassifier(estimator=clf,learning_rate=0.2,n_estimators=100)
adb.fit(x_train,y_train)

print("實際:",y_test)
print("預測結果:",adb.predict(x_test))
print("準確率:",adb.score(x_test,y_test))

plt.scatter(x[:,2],x[:,3],c=y)
plt.scatter(x[:,2],x[:,3],c=adb.predict(x))

plt.show()
