# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:41:27 2022

@author: Yanis Escartin
"""
#importing libraries 
import os
import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
import seaborn as sns
from dmba import stepwise_selection
from dmba import AIC_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score)

#setting the right directory
%cd Y:\Document\Yanis\Professionel\Portfolio\titanic

titanic = pd.read_csv('titanic.csv', sep=';')
titanic
titanic .info()

#checking for missing values
titanic.isna().sum()

#replacing string values
titanic['Sex'] = titanic['Sex'].replace('male', 1)
titanic['Sex'] = titanic['Sex'].replace('female', 0)
titanic['Embarked_P'] = titanic['Embarked_P'].replace('C', 1)
titanic['Embarked_P'] = titanic['Embarked_P'].replace('S', 2)
titanic['Embarked_P'] = titanic['Embarked_P'].replace('Q', 3)

titanic['Embarked_P'] = titanic['Embarked_P'].astype('int64')

titanic['Sex'] = titanic['Sex'].astype('int64')

#dropping non relevant columns
data = titanic[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'S_Ob', 'Pc_Ob', 'Embarked_P']]
data
data.info()

#splitting the data
t_train, t_test = train_test_split(data, test_size=0.20, random_state=42, shuffle=True)  

t_train.dropna(0,inplace=True)   
t_test.dropna(0,inplace=True) 

t_train
t_train.info()

t_test
t_test.info()

#first model: 
t_train1 = t_train.copy()
t_test1 = t_test.copy()

model_train1= smf.logit("Survived ~ Pclass + Sex + Age + Fare + S_Ob + Pc_Ob + Embarked_P", data=t_train1)
log_reg_train1=model_train1.fit()

log_reg_train1.summary()

log_reg_train1.summary2()

#building prediced column for both train and test 

yhat1 = log_reg_train1.predict(t_train1[['Pclass', 'Sex', 'Age', 'Fare', 'S_Ob', 'Pc_Ob', 'Embarked_P']])
t_train1['yhat']=yhat1
predict1 = log_reg_train1.predict(t_test1[['Pclass', 'Sex', 'Age', 'Fare', 'S_Ob', 'Pc_Ob', 'Embarked_P']])
t_test1['predict']=predict1

t_train1.loc[t_train1['yhat'] >= 0.6, ['status_hat']] = 1
t_train1.loc[t_train1['yhat'] < 0.6, ['status_hat']] = 0

t_test1.loc[t_test1['predict'] >= 0.6, ['predict_stat']] = 1
t_test1.loc[t_test1['predict'] < 0.6, ['predict_stat']] = 0

# model 1 training accuracy and test accuracy 

cm_t_train1 = confusion_matrix(t_train1.Survived, t_train1.status_hat) 
print ("Confusion Matrix : \n", cm_t_train1) 
  

print('Training accuracy Model 1 = ', accuracy_score(t_train1.Survived, t_train1.status_hat))

cm_t_test1 = confusion_matrix(t_test1.Survived, t_test1.predict_stat) 
print ("Confusion Matrix : \n", cm_t_test1) 
  

print('Test accuracy of model 1 = ', accuracy_score(t_test1.Survived, t_test1.predict_stat))

#second model
t_train2 = t_train.copy()
t_test2 = t_test.copy()

model_train2= smf.logit("Survived ~ Pclass + Sex + Age + S_Ob + Pc_Ob", data=t_train2)
log_reg_train2=model_train2.fit()

log_reg_train2.summary()

log_reg_train2.summary2()

yhat2 = log_reg_train2.predict(t_train2[['Pclass', 'Sex', 'Age', 'S_Ob', 'Pc_Ob']])

yhat2

t_train2['yhat']=yhat2

predict2 = log_reg_train2.predict(t_test2[['Pclass', 'Sex', 'Age', 'S_Ob', 'Pc_Ob']])

predict2

t_test2['predict']=predict2

t_train2.loc[t_train2['yhat'] >= 0.6, ['status_hat']] = 1
t_train2.loc[t_train2['yhat'] < 0.6, ['status_hat']] = 0

t_test2.loc[t_test2['predict'] >= 0.6, ['predict_stat']] = 1
t_test2.loc[t_test2['predict'] < 0.6, ['predict_stat']] = 0

# model 2 training accuracy and test accuracy 

cm_t_train2 = confusion_matrix(t_train2.Survived, t_train2.status_hat) 
print ("Confusion Matrix : \n", cm_t_train2) 
  

print('Training accuracy Model 2 = ', accuracy_score(t_train2.Survived, t_train2.status_hat))

cm_t_test2 = confusion_matrix(t_test2.Survived, t_test2.predict_stat) 
print ("Confusion Matrix : \n", cm_t_test2) 
  

print('Test accuracy of model 2 = ', accuracy_score(t_test2.Survived, t_test2.predict_stat))




