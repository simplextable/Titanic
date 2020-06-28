# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:38:13 2020

@author: Predator
"""

### BU ÇALIŞMADA 3 PARAMETRE VE DECISION TREE VE RF ILE CLASSIFICATION TAHMINLEMESİ YAPTIK


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam




from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV



df = pd.read_csv("test.csv")
print(df.info())
print(df.isnull().sum()) 





# Veri DÜZENLEME
  

df['Cabin']=df['Cabin'].fillna(0)
df.loc[df.Cabin.str[0]=="A", "Cabin"] = 1
df.loc[df.Cabin.str[0]=="B", "Cabin"] = 2
df.loc[df.Cabin.str[0]=="C", "Cabin"] = 3
df.loc[df.Cabin.str[0]=="D", "Cabin"] = 4
df.loc[df.Cabin.str[0]=="E", "Cabin"] = 5
df.loc[df.Cabin.str[0]=="F", "Cabin"] = 6
df.loc[df.Cabin.str[0]=="G", "Cabin"] = 7



#sex = df.iloc[:,3:4].values
#print(sex)
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#sex[:,0] = le.fit_transform(sex[:,0])
#print(sex)
#sex = pd.DataFrame(data = sex, index = range(891), columns=["sex"] )
#df= pd.concat([sex,df],axis=1)
#df.drop(['Sex'],axis=1,inplace=True)




#df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])      # Tek Tek Düzeltmek Stringler
#emberked = df.iloc[:,11:12].values
#from sklearn.preprocessing import LabelEncoder
#le2 = LabelEncoder()
#emberked[:,0] = le2.fit_transform(emberked[:,0])
#emberked = pd.DataFrame(data = emberked, index = range(891), columns=["emberked"] )
#df= pd.concat([emberked,df],axis=1)
#df.drop(['Embarked'],axis=1,inplace=True)




import re as re
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""


df['Title'] = df['Name'].apply(get_title)

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df['Title'] = df['Title'].map(title_mapping)





df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)




df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0]) 
df['Embarked'] = df['Embarked'].map( {'S': 0, 'Q': 1,"C":2} ).astype(int)
    



   
    
#df['CategoricalAge'] = pd.cut(df['Age'], 5)


age_avg 	   = df['Age'].mean()
age_std 	   = df['Age'].std()
age_null_count = df['Age'].isnull().sum()
    
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
df['Age'][np.isnan(df['Age'])] = age_null_random_list
df['Age'] = df['Age'].astype(int)


df.loc[ df['Age'] <= 16, 'Age']= 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age']= 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age']= 3
df.loc[ df['Age'] > 64, 'Age']= 4



df.drop(['PassengerId',"Fare","Ticket","Name"],axis=1,inplace=True)

df.to_csv('testDF.csv',index=False)