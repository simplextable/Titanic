import numpy as np
import pandas as pd
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML
# collection of machine learning algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Common Model Helpers
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import model_selection
import pylab as pl
from sklearn.metrics import roc_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.metrics import confusion_matrix



pd.set_option("display.max_rows",2200)  # KISALTMA ENGELLEME
pd.set_option("display.max_columns",2200)  # KISALTMA ENGELLEME

train = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")

train.drop(["Cabin", "PassengerId", "Name", "Sex", "Ticket", "Embarked"],axis=1,inplace=True)


train = train.dropna()

y = train[["Survived"]]

train.drop(["Survived"],axis=1,inplace=True)

print(train.isnull().sum()) 




from sklearn.model_selection  import train_test_split
x_train, x_test ,y_train, y_test = train_test_split(train ,y , test_size = 0.33 , random_state = 0)




from sklearn.ensemble import RandomForestClassifier

R_F = RandomForestClassifier()
R_F.fit(x_train, y_train)

y_pred = R_F.predict(x_test)

print("Random Forest R2")
print(confusion_matrix(y_test, y_pred))






