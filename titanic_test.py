# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:44:25 2020

@author: Predator
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


#veriler = pd.read_csv("train.csv")
veriler = pd.read_csv("test.csv")

veriler2 = veriler.copy() 
veriler2.drop(['PassengerId', "Pclass", "Name", "Ticket","Fare","Cabin", "Embarked","Sex"],axis=1,inplace=True)

veriler2["Age"] = veriler2["Age"].fillna(veriler2["Age"].mean())


print(veriler_test.isnull().sum())

sns.heatmap(veriler2.isnull(),yticklabels=False,cbar=False)

sex = veriler[["Sex"]]


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
sex=ohe.fit_transform(sex).toarray()

sex = pd.DataFrame(data = sex, index = range(418), columns=["Female","Male"] )
sex.drop(["Female"],axis=1,inplace=True)

veriler3= pd.concat([veriler2,sex],axis=1)

veriler3.to_csv('titanictest.csv',index=False)