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
x = np.random.standard_normal(size=(p,n)).reshape(20,1000)
beta = np.random.standard_normal(p)
eps = np.random.standard_normal(n)
beta[3] = 0
beta[4] = 0
beta[9] = 0
beta[19] = 0
beta[10] = 0

y = np.dot(beta, x) + eps

data = pd.DataFrame()
betas = pd.Series()
data['y'] = y
xcols = []

for i in range(0, p):

    xcols.append('x' + str(i))
    
    data[xcols[-1]] = x[i]
    betas[xcols[-1]] = beta[i]

#Keep test set#
data_train = data.sample(frac=0.1, random_state=777)
data_test = data.drop(data_train.index)
    
##Best models by selection on y1##
best_models = lab1.best_subset(data_train, x=xcols, y='y', nsplits=10)

print('\nLowest CV error best subset model:')
very_best = best_models.loc[best_models['CVerr'].argmin()]
print(very_best)
print('\nBest model linear regression parameters')
print(zip(very_best['Vars'].split(';'), very_best['model'].coef_))
print('\nSimulation linear regression parameters')
print(betas[very_best['Vars'].split(';')])
print('\n Coef. Squared error:')
print(np.sqrt(np.sum(np.square(betas[very_best['Vars'].split(';')] - very_best['model'].coef_))))

test_mse = []
coef_se = []

for itest in best_models.index :
    combs = best_models.loc[itest, 'Vars'].split(';')
    X_test = np.array(data_test[combs])
    Y_test = np.array(data_test['y'])
    Y_test_pred = best_models.loc[itest, 'model'].predict(X_test)
    test_mse.append(mean_squared_error(Y_test_pred, Y_test))
    fit_coef = best_models.loc[itest, 'model'].coef_
    coef_se.append(np.sqrt(np.sum(np.square(betas[combs] - fit_coef))))
    
fbest, ((axbest1, axbest2)) = plt.subplots(1, 2)
axbest1.plot(best_models['NumVar'], best_models['MSE'], label='Training')
axbest1.plot(best_models['NumVar'], test_mse, label='Test')
axbest1.plot(best_models['NumVar'], best_models['CVerr'], label='CV')
axbest1.set_xlabel('k')
axbest1.set_ylabel('MSE')
axbest1.legend(loc='best')
axbest2.plot(best_models['NumVar'], coef_se)
axbest2.set_xlabel('k')
axbest2.set_ylabel('Coef. squared error')
fbest.suptitle('Best subset selection')
plt.tight_layout
plt.show()
