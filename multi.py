# -*- coding: utf-8 -*-
"""


@author: Kaival Shah
"""

#importing all the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#state is categorical features we need to 
#convert in to one hot enconding

states=pd.get_dummies(X['State'],drop_first=True)

#dropping the state and adding hot encodded 
#variable states 
X=X.drop('State',axis=1)

X=pd.concat([X,states],axis=1)

#train , test split 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#fitting multiple regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results 

y_pred=regressor.predict(X_test)

#r^2 value 
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

