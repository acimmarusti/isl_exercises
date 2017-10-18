from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import chap6lab1 as lab1

#Simulated Data#
np.random.seed(1)
p = 20
n = 1000
x = np.random.standard_normal(size=(n,p))
eps = np.random.standard_normal(p)
beta = np.random.standard_normal(p)
beta[3] = 0
beta[4] = 0
beta[9] = 0
beta[19] = 0
beta[10] = 0

y = beta * x + eps

data = pd.DataFrame()
data['y'] = y

for i in range(0, p):

    xcols.append('x' + str(i))
    
    data[xcols[-1]] = x[i]

#Keep test set#
X_train, X_test, Y_train, Y_test = train_test_split(data[xcols], data['y'], test_size=0.9, random_state=42)
    
##Best models by selection on y1##
best_models = lab1.best_subset(data, x=xcols, y='y', nsplits=10)

print('\nLowest CV error best subset model 1:')
print(best_models.loc[best_models['CVerr'].argmin()])

print(best_models.head())
