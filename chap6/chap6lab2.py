from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

filename = '../Hitters.csv'

#Load raw data to pandas dataframe#
data = pd.read_csv(filename).dropna()

#Binarize columns#
data['LeagueN'] = (data['League'] == 'N').astype(int)
data['DivisionW'] = (data['Division'] == 'W').astype(int)
data['NewLeagueN'] = (data['NewLeague'] == 'N').astype(int)

#Only numerical cols#
colmask = data.dtypes != object
numcols = colmask.index[colmask == True]
xcols = list(numcols)
xcols.remove('Salary')

#Alphas#
alphas = 10**np.linspace(10,-2,100)*0.5

ridge = Ridge(normalize=True)
coefs = []

X = np.array(data[xcols])
Y = np.array(data['Salary'])

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, Y)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

plt.show()
