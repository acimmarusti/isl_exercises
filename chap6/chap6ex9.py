from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

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

###Linear Regression###

lreg = LinearRegression()
lreg.fit(X_train, Y_train)

print('\nSimple Linear Regression MSE:')
print(mean_squared_error(Y_test, lreg.predict(X_test)))

#10-fold Cross-validation object#
kfcv = KFold(n_splits=10, shuffle=True, random_state=2)

###Ridge regression###
#Alphas#
alphas = 10**np.linspace(10,-2,100)*0.5

#RidgeCV with 10-fold cross-validation(similar to ISLR)#
rcv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True, cv=kfcv)
rcv.fit(X_train, Y_train)

print('\nBest RidgeCV alpha value:')
print(rcv.alpha_)

#Ridge regression using best alpha#
rbest = Ridge(alpha=rcv.alpha_, normalize=True)
rbest.fit(X_train, Y_train)

print('\nBest Ridge MSE:')
print(mean_squared_error(Y_test, rbest.predict(X_test)))

###Lasso regression###

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

###Principal components regression###

pca = PCA()
X_train_red = pca.fit_transform(scale(X_train))
n = len(X_train_red)

#Linear regression#
plreg = LinearRegression()

#MSE initialization to only intercept#
scores_intercept = cross_val_score(plreg, np.zeros((n,1)), y=Y_train, scoring='neg_mean_squared_error', cv=kfcv)
mse_pcr = [-np.mean(scores_intercept)]
mse_pcr_std = [np.std(scores_intercept)]

for ii in range(1, len(xcols) + 1):
    scores = cross_val_score(plreg, X_train_red[:,:ii], y=Y_train, scoring='neg_mean_squared_error', cv=kfcv)
    mse_pcr.append(-np.mean(scores))
    mse_pcr_std.append(np.std(scores))

npc = mse_pcr.index(min(mse_pcr))

print('\nPCR components to use:')
print(npc)

#Test set prediction#
X_test_red = pca.transform(scale(X_test))[:,:npc + 1]
nlreg = LinearRegression()
nlreg.fit(X_train_red[:,:npc + 1], Y_train)
Y_test_pred = nlreg.predict(X_test_red)

print('\nPCR Test set MSE:')
print(mean_squared_error(Y_test, Y_test_pred))

###PLS###

mse_pls = []
mse_pls_std = []

for ii in range(1, len(xcols) + 1):
    pls = PLSRegression(n_components=ii)
    scores_pls = cross_val_score(pls, scale(X_train), y=Y_train, scoring='neg_mean_squared_error', cv=kfcv)
    mse_pls.append(-np.mean(scores_pls))
    mse_pls_std.append(np.std(scores_pls))

npls = mse_pls.index(min(mse_pls))

print('\nPLS components to use:')
print(npls)

pls = PLSRegression(n_components=npls)
pls.fit(scale(X_train), Y_train)

print('\nPLS Test set MSE:')
print(mean_squared_error(Y_test, pls.predict(scale(X_test))))
