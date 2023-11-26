import numpy as np
import pandas as pd
data = pd.read_excel('iris .xls')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree  import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score

import pickle

X = data.drop('Classification',axis=1)
y = data['Classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Logistic Regression
model = LogisticRegression()
model.fit(X_train,y_train)
model_lr_pred = model.predict(X_test)
print('Logistic Regression accuracy: ',model.score(X_test,y_test)*100)

accuracy_score(model_lr_pred,y_test)

#KNN
model = KNeighborsClassifier()
model.fit(X_train,y_train)
model_knn_pred = model.predict(X_test)
print('KNN accuracy : ',model.score(X_test,y_test)*100)

accuracy_score(model_knn_pred,y_test)

#SVM
model = SVC(kernel='linear')
model.fit(X_train,y_train)
model_svm_pred = model.predict(X_test)
print('SVM accuracy : ',model.score(X_test,y_test)*100)

#Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
model_dt_pred = model.predict(X_test)
print('DT accuracy :', model.score(X_test,y_test)*100)

accuracy_score(model_dt_pred,y_test)

#save the model

pickle_file = 'iris_model.pickle'
with open(pickle_file,'wb') as file:
    pickle.dump(model,file)



