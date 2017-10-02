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
x = np.random.standard_normal(100)
eps = np.random.standard_normal(100)
y1 = 1 + 2 * x - 5 * np.power(x, 2) + 3 * np.power(x, 3) + eps
y2 = 3 + 9 * np.power(x, 7) + eps

data = pd.DataFrame()
data['y1'] = y1
data['y2'] = y2
data['x'] = x

xcols = ['x']

for i in range(2, 11):

    xcols.append('x' + str(i))
    
    data[xcols[-1]] = np.power(x, i)
    
##Best models by selection on y1##
best_models1 = lab1.best_subset(data, x=xcols, y='y1', nsplits=10)

print('\nLowest CV error best subset model 1:')
print(best_models1.loc[best_models1['CVerr'].argmin()])

best_models2 = lab1.best_subset(data, x=xcols, y='y2', nsplits=10)

print('\nLowest CV error best subset model 2:')
print(best_models2.loc[best_models2['CVerr'].argmin()])

fwd_models1 = lab1.forward_sel(data, x=xcols, y='y1', nsplits=10)
    
print('\nLowest CV error best forward model 1:')
print(fwd_models1.loc[fwd_models1['CVerr'].argmin()])

back_models1 = lab1.backward_sel(data, x=xcols, y='y1', nsplits=10)

print('\nLowest CV error best backward model 1:')
print(back_models1.loc[back_models1['CVerr'].argmin()])


##Lasso regression with cross-validation##

#Alphas#
alphas = 10**np.linspace(10,-2,100)*0.5

X = np.array(data[xcols])
Y1 = np.array(data['y1'])
Y2 = np.array(data['y2'])

#Full Lasso regression#
lasso1 = Lasso(max_iter=10000, normalize=True)
coefs1 = []

for a in alphas:
    lasso1.set_params(alpha=a)
    lasso1.fit(scale(X), Y1)
    coefs1.append(lasso1.coef_)

axl1 = plt.gca()
axl1.plot(alphas*2, coefs1)
axl1.set_xscale('log')
axl1.set_title('Lasso model 1')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

#10-fold Cross-validation object#
kfcv = KFold(n_splits=10)

#LassoCV with 10-fold cross-validation(similar to ISLR)#
lcv1 = LassoCV(alphas=None, max_iter=100000, normalize=True, cv=kfcv, n_jobs=2)
lcv1.fit(scale(X), Y1)
lcv2 = LassoCV(alphas=None, max_iter=100000, normalize=True, cv=kfcv, n_jobs=2)
lcv2.fit(scale(X), Y2)

print('\nBest LassoCV alpha value model 1:')
print(lcv1.alpha_)
print('\nBest LassoCV alpha value model 2:')
print(lcv2.alpha_)

#Lasso regression using best alpha#
lbest1 = Lasso(alpha=lcv1.alpha_, normalize=True)
lbest1.fit(scale(X), Y1)
lbest2 = Lasso(alpha=lcv2.alpha_, normalize=True)
lbest2.fit(scale(X), Y2)

print('\nBest Lasso model 1 MSE:')
print(mean_squared_error(Y1, lbest1.predict(scale(X))))
print('\nBest Lasso model 2 MSE:')
print(mean_squared_error(Y2, lbest2.predict(scale(X))))

print('\nLasso Coeficients model 1:')
print(pd.Series(lbest1.coef_, index=xcols))
print('\nLasso Coeficients model 2:')
print(pd.Series(lbest2.coef_, index=xcols))

##Plots best models on y1##
fbest, ((axbest1, axbest2), (axbest3, axbest4), (axbest5, axbest6)) = plt.subplots(3, 2, sharex='col')
axbest1.plot(best_models1['NumVar'], best_models1['AdjR2'])
axbest1.set_ylabel('Adjusted R2')
axbest2.plot(best_models1['NumVar'], best_models1['AIC'])
axbest2.set_ylabel('AIC')
axbest3.plot(best_models1['NumVar'], best_models1['Cp'])
axbest3.set_ylabel('Cp')
axbest4.plot(best_models1['NumVar'], best_models1['BIC'])
axbest4.set_ylabel('BIC')
axbest5.set_xlabel('k')
axbest6.plot(best_models1['NumVar'], best_models1['CVerr'])
axbest6.set_xlabel('k')
axbest6.set_ylabel('CV error')
fbest.suptitle('Best subset selection model 1')

ffwd, ((axfwd1, axfwd2), (axfwd3, axfwd4), (axfwd5, axfwd6)) = plt.subplots(3, 2, sharex='col')
axfwd1.plot(fwd_models1['NumVar'], fwd_models1['AdjR2'])
axfwd1.set_ylabel('Adjusted R2')
axfwd2.plot(fwd_models1['NumVar'], fwd_models1['AIC'])
axfwd2.set_ylabel('AIC')
axfwd3.plot(fwd_models1['NumVar'], fwd_models1['Cp'])
axfwd3.set_ylabel('Cp')
axfwd4.plot(fwd_models1['NumVar'], fwd_models1['BIC'])
axfwd4.set_ylabel('BIC')
axfwd5.set_xlabel('k')
axfwd6.plot(fwd_models1['NumVar'], fwd_models1['CVerr'])
axfwd6.set_xlabel('k')
axfwd6.set_ylabel('CV error')
ffwd.suptitle('Forward subset selection model 1')

fback, ((axback1, axback2), (axback3, axback4), (axback5, axback6)) = plt.subplots(3, 2, sharex='col')
axback1.plot(back_models1['NumVar'], back_models1['AdjR2'])
axback1.set_ylabel('Adjusted R2')
axback2.plot(back_models1['NumVar'], back_models1['AIC'])
axback2.set_ylabel('AIC')
axback3.plot(back_models1['NumVar'], back_models1['Cp'])
axback3.set_ylabel('Cp')
axback4.plot(back_models1['NumVar'], back_models1['BIC'])
axback4.set_ylabel('BIC')
axback5.set_xlabel('k')
axback6.plot(back_models1['NumVar'], back_models1['CVerr'])
axback6.set_xlabel('k')
axback6.set_ylabel('CV error')
fback.suptitle('Backward subset selection model 1')

plt.tight_layout
plt.show()
