import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

test_data= pd.read_csv('test.csv')
train_data= pd.read_csv('train.csv')
#gender_submissions=pd.read_csv('gender_submission.csv'


y=train_data[["Survived"]]
y=pd.DataFrame(train_data.Survived)

##numpy dizileri dataframe donusumu
#sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
#print(sonuc)
#
#print(y)
sns.set_style("whitegrid")
missing = train_data.isnull().sum()
missing = missing[missing > 0]/891
missing.sort_values(inplace=True)
missing.plot.bar()