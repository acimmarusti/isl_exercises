from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression, PLSSVD
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

#Full data split#
X = np.array(data[xcols])
Y = np.array(data['Salary'])

#Keep test set#
X_train, X_test, Y_train, Y_test = train_test_split(data[xcols], data['Salary'], test_size=0.5, random_state=42)

###Principal components regression###
pca = PCA()
X_red = pca.fit_transform(scale(X))
n = len(X_red)


#10-fold Cross-validation object#
kfcv = KFold(n_splits=10, shuffle=True, random_state=2)

#Linear regression#
lreg = LinearRegression()

#MSE initialization to only intercept#
scores_intercept = cross_val_score(lreg, np.zeros((n,1)), y=Y, scoring='neg_mean_squared_error', cv=kfcv)
mse = [-np.mean(scores_intercept)]
mse_std = [np.std(scores_intercept)]

for ii in range(1, len(xcols) + 1):
    scores = cross_val_score(lreg, X_red[:,:ii], y=Y, scoring='neg_mean_squared_error', cv=kfcv)
    mse.append(-np.mean(scores))
    mse_std.append(np.std(scores))

plt.figure()
plt.plot(mse, '-v')
plt.title('PCR')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE mean')

plt.figure()
plt.plot(mse_std, '-v')
plt.title('PCR')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE Stdev')
plt.tight_layout()

"""
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
"""
plt.show()
