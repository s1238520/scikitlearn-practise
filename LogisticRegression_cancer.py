from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

breast_cancer=load_breast_cancer()
x=breast_cancer.data
y=breast_cancer.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

clf1=LogisticRegression()
clf1.fit(x_train,y_train)

for f,w in zip(breast_cancer.feature_names,clf1.coef_[0]):
    print("{0:<23}:{1:6.2f}".format(f,w))

print("實際:",y_test)
print("預測:",clf1.predict(x_test))
print("準確率:",clf1.score(x_test,y_test))

plt.scatter(x[:,2],x[:,3],c=y)
plt.show()


