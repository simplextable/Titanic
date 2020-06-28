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
veriler_test = pd.read_csv("test.csv")

veriler2 = veriler.copy() 
veriler2.drop(['PassengerId', "Pclass", "Name","SibSp", "Ticket","Fare","Cabin","Embarked", "Survived","Sex"],axis=1,inplace=True)

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


##Verilerin Ayrılması
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(veriler3, survived, test_size = 0.33, random_state=0 )





## RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print("Random Forest")
print(cm)    
"""
accuracy = 0,77
"""

## DECISION TREE
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = "entropy")
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print("Decions Tree")
print(cm)    
"""
accuracy = 0,745
"""


