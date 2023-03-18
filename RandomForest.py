from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
rfc=RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =0, min_samples_leaf = 5)
rfc.fit(X_train,y_train)
y_predict=rfc.predict(X_test)
y_predict
rfc.score(X_test,y_test)
print("實際:",y_test,"\n預測結果:",y_predict)
print("準確率:",rfc.score(X_test,y_test))

imp=rfc.feature_importances_
print("特徵重要性:",imp)

plt.scatter(X[:,2],X[:,3],c=y)
plt.show()