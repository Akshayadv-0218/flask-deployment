import numpy as np
import pandas as pd
data = pd.read_excel('iris .xls')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,precision_score,recall_score

import pickle

X = data.drop('Classification',axis=1)
y = data['Classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Logistic Regression
lr = LogisticRegression()
model = lr.fit(X_train,y_train)
lr_pred = model.predict(X_test)
print('Logistic Regression accuracy: ',accuracy_score(y_test,lr_pred))



#save the model

pickle_file = 'iris_model.pickle'
with open(pickle_file,'wb') as file:
    pickle.dump(model,file)



