# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:10:57 2020

@author: Predator
"""

### MÜMKÜN OLAN TÜM VERİLERİN EKLENMESİ İLE BULUNAN SONUÇ

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


veriler = pd.read_csv("train.csv")

veriler2 = veriler.copy() 
veriler2.drop(['PassengerId', "Name","Survived", "Sex", "Cabin", "Ticket", "Pclass"],axis=1,inplace=True)

## EKSİK VERİLERİN DOLDURULMASI
veriler2["Age"] = veriler2["Age"].fillna(veriler2["Age"].mean())
veriler2['Embarked']=veriler2['Embarked'].fillna(veriler2['Embarked'].mode()[0])


survived = veriler["Survived"]
sex = veriler[["Sex"]]
embarked = veriler2[["Embarked"]]
pclass = veriler[["Pclass"]]

veriler2.drop(["Embarked"],axis=1,inplace=True)


## KATEGORİK VERİLERİN DÖNÜŞTÜRÜLMESİ
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
sex=ohe.fit_transform(sex).toarray()


ohe = OneHotEncoder(categorical_features='all')
embarked=ohe.fit_transform(embarked).toarray()

ohe = OneHotEncoder(categorical_features='all')
pclass=ohe.fit_transform(pclass).toarray()



## DÖNÜŞTÜRÜLERN VERİLERİN TEKRAR EKLENMESİ
sex = pd.DataFrame(data = sex, index = range(891), columns=["Female","Male"] )
sex.drop(["Female"],axis=1,inplace=True)
veriler3= pd.concat([veriler2,sex],axis=1)

embarked = pd.DataFrame(data = embarked, index = range(891), columns=["S","C","Q"])
veriler3= pd.concat([veriler2,embarked],axis=1)

pclass= pd.DataFrame(data = pclass, index = range(891), columns=["First","Second","Third"])
veriler3= pd.concat([veriler3,pclass],axis=1)



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
accuracy = 0,703
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
accuracy = 0,68
"""
