from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

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

#Full Ridge regression#
X = np.array(data[xcols])
Y = np.array(data['Salary'])

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, Y)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_title('Ridge')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

##Ridge regression with cross-validation##

#Keep test set#
X_train, X_test, Y_train, Y_test = train_test_split(data[xcols], data['Salary'], test_size=0.5, random_state=42)

#10-fold Cross-validation object#
kfcv = KFold(n_splits=10)

#RidgeCV with 10-fold cross-validation(similar to ISLR)#
#rcv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True)
rcv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True, cv=kfcv)
rcv.fit(X_train, Y_train)

print('\nBest RidgeCV alpha value:')
print(rcv.alpha_)

#Ridge regression using best alpha#
rbest = Ridge(alpha=rcv.alpha_, normalize=True)
rbest.fit(X_train, Y_train)

print('\nBest Ridge MSE:')
print(mean_squared_error(Y_test, rbest.predict(X_test)))

print('\nRidge Coeficients:')
print(pd.Series(rbest.coef_, index=xcols))

#Full Lasso regression#
lasso = Lasso(max_iter=10000, normalize=True)
coefs2 = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X), Y)
    coefs2.append(lasso.coef_)

ax2 = plt.gca()
ax2.plot(alphas*2, coefs)
ax2.set_xscale('log')
ax2.set_title('Lasso')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

##Lasso regression with cross-validation##

#LassoCV with 10-fold cross-validation(similar to ISLR)#
lcv = LassoCV(alphas=None, max_iter=100000, normalize=True, cv=kfcv, n_jobs=2)
lcv.fit(X_train, Y_train)

print('\nBest LassoCV alpha value:')
print(lcv.alpha_)

#Ridge regression using best alpha#
lbest = Lasso(alpha=lcv.alpha_, normalize=True)
lbest.fit(X_train, Y_train)

print('\nBest Lasso MSE:')
print(mean_squared_error(Y_test, lbest.predict(X_test)))

print('\nLasso Coeficients:')
print(pd.Series(lbest.coef_, index=xcols))

plt.show()
