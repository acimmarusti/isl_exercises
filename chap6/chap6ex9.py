from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import chap6lab1 as lab1

#Loading Data#
filename = '../College.csv'

rawdata = pd.read_csv(filename)

data = rawdata.rename(columns={rawdata.columns[0]: 'college'})

#Binarize columns#
data['PrivateY'] = (data['Private'] == 'Yes').astype(int)

#Only numerical cols#
colmask = data.dtypes != object
numcols = colmask.index[colmask == True]
xcols = list(numcols)
xcols.remove('Apps')

#Keep test set#
X_train, X_test, Y_train, Y_test = train_test_split(data[xcols], data['Apps'], test_size=0.5, random_state=42)
    
lreg = LinearRegression()
lreg.fit(X_train, Y_train)

print('\nLinear Regression MSE:')
print(mean_squared_error(Y_test, lreg.predict(X_test)))

