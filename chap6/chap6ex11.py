from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import chap6lab1 as lab1

#Load boston dataset from sklearn#
boston = load_boston()

#Columns#
#print(boston['feature_names'])
#Descriptio#
#print(boston['DESCR'])

rawdata = pd.DataFrame(boston.data, columns=boston.feature_names)
rawdata['MEDV'] = boston.target

allcols = rawdata.columns
xcols = list(allcols)
xcols.remove('CRIM')

#Keep test set#
data_train = rawdata.sample(frac=0.5, random_state=777)
data_test = rawdata.drop(data_train.index)
    
##Best subset selection##
best_models = lab1.best_subset(data_train, x=xcols, y='CRIM', nsplits=10)

print('\nLowest CV error best subset model:')
very_best = best_models.loc[best_models['CVerr'].argmin()]
print(very_best)

#Train set#
X_train = np.array(data_train[xcols])
Y_train = np.array(data_train['CRIM'])

#Test set#
X_test = np.array(data_test[xcols])
Y_test = np.array(data_test['CRIM'])

combs = very_best['Vars'].split(';')
X_test_vb = np.array(data_test[combs])
Y_pred_vb = very_best['model'].predict(X_test_vb)
vb_mse = mean_squared_error(Y_test, Y_pred_vb)

print('\nBest subset Test MSE:')
print(vb_mse)

##Ridge regression with cross-validation##

#Alphas#
alphas = 10**np.linspace(10,-2,100)*0.5

#10-fold Cross-validation object#
kfcv = KFold(n_splits=10)

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

print('\nRidge Coeficients:')
print(pd.Series(rbest.coef_, index=xcols))


##Lasso regression with cross-validation##

#LassoCV with 10-fold cross-validation(similar to ISLR)#
lcv = LassoCV(alphas=None, max_iter=100000, normalize=True, cv=kfcv, n_jobs=4)
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
lreg = LinearRegression()

#MSE initialization to only intercept#
scores_intercept = cross_val_score(lreg, np.zeros((n,1)), y=Y_train, scoring='neg_mean_squared_error', cv=kfcv)
mse = [-np.mean(scores_intercept)]
mse_std = [np.std(scores_intercept)]

for ii in range(1, len(xcols) + 1):
    scores = cross_val_score(lreg, X_train_red[:,:ii], y=Y_train, scoring='neg_mean_squared_error', cv=kfcv)
    mse.append(-np.mean(scores))
    mse_std.append(np.std(scores))

#Test set prediction#
npc = 4
X_test_red = pca.transform(scale(X_test))[:,:npc + 1]
nlreg = LinearRegression()
nlreg.fit(X_train_red[:,:npc + 1], Y_train)
Y_test_pred = nlreg.predict(X_test_red)

print('\nPCR Test set MSE:')
print(mean_squared_error(Y_test, Y_test_pred))

#PCR plots#
plt.figure()
plt.plot(mse, '-v')
plt.title('PCR')
plt.xlabel('Number of principal components in regression')
plt.ylabel('Train MSE mean')
plt.tight_layout()

plt.figure()
plt.plot(mse_std, '-v')
plt.title('PCR')
plt.xlabel('Number of principal components in regression')
plt.ylabel('Train MSE Stdev')
plt.tight_layout()

#plt.show()
