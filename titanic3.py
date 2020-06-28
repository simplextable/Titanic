# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:59:56 2020

@author: Predator
"""

### BU ÇALIŞMADA 3 PARAMETRE VE DECISION TREE VE RF ILE CLASSIFICATION TAHMINLEMESİ YAPTIK


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


veriler = pd.read_csv("train.csv")
veriler_test = pd.read_csv("titanictest.csv")
gender_submission = pd.read_csv("gender_submission.csv")


veriler2 = veriler.copy() 
veriler2.drop(['PassengerId', "Pclass", "Name", "Ticket","Fare","Cabin", "Embarked", "Survived","Sex"],axis=1,inplace=True)

survived = veriler["Survived"]

veriler2["Age"] = veriler2["Age"].fillna(veriler2["Age"].mean())


print(veriler_test.isnull().sum())

sns.heatmap(veriler_test.isnull(),yticklabels=False,cbar=False)



## DATA KATEGORİLEŞTİRME

sex = veriler[["Sex"]]


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
sex=ohe.fit_transform(sex).toarray()

sex = pd.DataFrame(data = sex, index = range(891), columns=["Female","Male"] )
sex.drop(["Female"],axis=1,inplace=True)

veriler3= pd.concat([veriler2,sex],axis=1)


###Verilerin Ayrılması
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(veriler3, survived, test_size = 0.33, random_state=0 )





## RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 300, max_depth = 5, criterion = "gini", min_samples_leaf = 2)
rfc.fit(veriler3, survived)
y_pred = rfc.predict(veriler_test)


gender_submission.drop(["Survived"],axis=1,inplace=True)


y_pred = pd.DataFrame(data = y_pred, index = range(418), columns=["Survived"]) 


son_dosya = pd.concat([gender_submission, y_pred],axis=1)

son_dosya.to_csv('result.csv',index=False)

#cm = confusion_matrix(y_test,y_pred)
#print("Random Forest")
#print(cm)    
#"""
#accuracy = 0,762
#acc = 0,796 (siblings eklenince)
#acc = 0,81 (siblings ve hyperparameter)
#"""

### DECISION TREE
#from sklearn.tree import DecisionTreeClassifier
#
#dtc = DecisionTreeClassifier(criterion = "entropy")
#dtc.fit(x_train,y_train)
#y_pred = dtc.predict(x_test)
#
#cm = confusion_matrix(y_test,y_pred)
#print("Decions Tree")
#print(cm)    
#"""
#accuracy = 0,745
#acc = 0,742 (siblings eklenince) 
#acc = 0,742 (siblings ve hyperparameter)
#
#"""
#
### LOGISTIC REGRESSION
#from sklearn.linear_model import LogisticRegression
#
#logr = LogisticRegression(random_state=0)
#logr.fit(x_train,y_train)
#
#y_pred = logr.predict(x_test)
#cm = confusion_matrix(y_test,y_pred)
#print("Logistic Regression")
#print(cm)    
#
#"""
#acc = 0,783 (siblingli)
#"""
#
#
