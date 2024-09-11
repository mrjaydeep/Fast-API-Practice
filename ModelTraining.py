# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:49:12 2024

@author: mrjay
"""

import numpy as np
import pandas as pd
import pickle
df=pd.read_csv('BankNote_Authentication.csv')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

### Train Test Split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

## Prediction
y_pred=classifier.predict(X_test)

### Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score

### Creating a Pickle file using serialization 

pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


classifier.predict([[2,3,4,1]])