from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

breast_cancer=load_breast_cancer()
x=breast_cancer.data
y=breast_cancer.target
clf1=LogisticRegression()
clf1.fit(x,y)
for f,w in zip(breast_cancer.feature_names,clf1.coef_[0]):
    print("{0:<23}:{1:6.2f}".format(f,w))
    
clf2=DecisionTreeClassifier(max_depth=2)
clf2.fit(x,y)
for f,w in zip(breast_cancer.feature_names,clf1.coef_[0]):
    print("{0:<23}:{1:6.2f}".format(f,w))