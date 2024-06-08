#breast cancer supervised learning using support vector machine learning
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd

#pull the breast cancer data directly from the web

cancer_data = datasets.load_breast_cancer()

x = cancer_data.data
y = cancer_data.target
#print the data
print(cancer_data['target'])
#split the data
x_train,x_test, y_train, y_test= train_test_split(cancer_data.data, cancer_data.target, test_size=0.4, random_state=209) 

#define sthe classifier support vector machine (SVM)
cls =svm.svc(kernal='linear')
#cls = svm.SVC(kernal="rbf", c = 0.1, gamma = 0.1 )
#Apply the fitting
cls.fit(x_train,y_train) 

#perform the prediction
pred = cls.predict(x_test)

#print all the results
print("accuracy:", metrics.accuracy_score(y_test,y_pred=pred))
print("precision:", metrics.precision_score(y_test,y_pred=pred))
print("recall", metrics.precision_score(y_test,y_pred=pred))
print(metrics.classification_report(y_test,y_pred=pred))

