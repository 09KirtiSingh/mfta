import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px

import pickle

df1=pd.read_csv("prevalence-by-mental-and-substance-use-disorder _AI.csv")
df2=pd.read_csv("mental-and-substance-use-as-share-of-disease -AI.csv")

data=pd.merge(df1,df2)

data.isnull().sum()
data.drop('Code',axis=1,inplace=True)

data.columns = ['Country', 'Year', 'Schizophrenia', 'Bipolar_Disorder', 'Eating_Disorder', 'Anxiety', 'Drug_Usage', 'Depression', 'Alcohol_Consumption', 'Mental_Fitness']


df=data
df.info()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
  if df[i].dtype=='object':
    df[i]=l.fit_transform(df[i])

X=df.drop('Mental_Fitness',axis=1)
Y=df['Mental_Fitness']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=.20,random_state=2)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(xtrain,ytrain)
ytrain_pred=rf.predict(xtrain)
mse=mean_squared_error(ytrain,ytrain_pred)
rmse=(np.sqrt(mean_squared_error(ytrain,ytrain_pred)))
r2=r2_score(ytrain,ytrain_pred)

print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 Score is {}'.format(r2))

ytest_pred=rf.predict(xtest)
mse=mean_squared_error(ytest,ytest_pred)
rmse=(np.sqrt(mean_squared_error(ytest,ytest_pred)))
r2=r2_score(ytest,ytest_pred)

print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 Score is {}'.format(r2))

#saving model to disc
pickle.dump(rf, open('model.pkl','wb'))

#loading model
model=pickle.load('model.pkl','rb')

model_score_r1 = load_model.score(xtest, ytest)