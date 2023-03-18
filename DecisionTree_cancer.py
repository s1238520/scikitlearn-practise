from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

breast_cancer=load_breast_cancer()
x=breast_cancer.data
y=breast_cancer.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

clf2=DecisionTreeClassifier(max_depth=2)
clf2.fit(x_train,y_train)


print("實際:",y_test)
print("預測:",clf2.predict(x_test))
print("準確率:",clf2.score(x_test,y_test))

plt.scatter(x[:,2],x[:,3],c=y)
plt.show()
