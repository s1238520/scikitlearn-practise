from sklearn.ensemble import BaggingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

iris=datasets.load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

clf=DecisionTreeClassifier()
#clf=LinearSVC()
#clf=GaussianNB()

bagging=BaggingClassifier(estimator=clf,n_estimators=20,
                          bootstrap=True,bootstrap_features=True,max_features=3,max_samples=0.5)

bagging.fit(x_train,y_train)
print("實際:",y_test)
print("預測結果:",bagging.predict(x_test))
print("準確率:",bagging.score(x_test,y_test))

plt.scatter(x[:,2],x[:,3],c=bagging.predict(x))
plt.show()



