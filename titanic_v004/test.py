import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


test_df = pd.read_csv("test.csv")
print(test_df.info())
print(test_df.isnull().sum()) 

  
test_df['Cabin']=test_df['Cabin'].fillna(0)
test_df.loc[test_df.Cabin.str[0]=="A", "Cabin"] = 1
test_df.loc[test_df.Cabin.str[0]=="B", "Cabin"] = 2
test_df.loc[test_df.Cabin.str[0]=="C", "Cabin"] = 3
test_df.loc[test_df.Cabin.str[0]=="D", "Cabin"] = 4
test_df.loc[test_df.Cabin.str[0]=="E", "Cabin"] = 5
test_df.loc[test_df.Cabin.str[0]=="F", "Cabin"] = 6
test_df.loc[test_df.Cabin.str[0]=="G", "Cabin"] = 7



import re as re
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
		if title_search:
		return title_search.group(1)
	return ""


test_df['Title'] = test_df['Name'].apply(get_title)
test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
test_df['Title'] = test_df['Title'].map(title_mapping)


test_df['Sex'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


test_df['Embarked']=test_df['Embarked'].fillna(test_df['Embarked'].mode()[0]) 
test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'Q': 1,"C":2} ).astype(int)
    

age_avg 	   = test_df['Age'].mean()
age_std 	   = test_df['Age'].std()
age_null_count = test_df['Age'].isnull().sum()
    
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
test_df['Age'][np.isnan(test_df['Age'])] = age_null_random_list
test_df['Age'] = test_df['Age'].astype(int)


test_df.loc[ test_df['Age'] <= 16, 'Age']= 0
test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1
test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age']= 2
test_df.loc[(test_df['Age'] > 48) & (test_df['Age'] <= 64), 'Age']= 3
test_df.loc[ test_df['Age'] > 64, 'Age']= 4

test_df.drop(['PassengerId',"Fare","Ticket","Name"],axis=1,inplace=True)

test_df.to_csv('test_df.csv',index=False)