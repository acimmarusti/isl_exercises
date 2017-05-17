from __future__ import print_function, division
import numpy as np
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_leverage_resid2

#Generate data#
np.random.seed(1)
x1 = np.random.uniform(size=100)
x2 = 0.5 * x1 + np.random.standard_normal(100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + np.random.standard_normal(100)

#Pandas dataframe#
data = pd.DataFrame()
data['x1'] = x1
data['x2'] = x2
data['y'] = y

#Regression x2 vs x1 (removing constant or intercept term)#
xlinreg = smf.ols(formula='x2 ~ x1 - 1', data=data).fit()
print('\n\nRegression result: x2 vs x1')
print(xlinreg.summary())

#plot x2 vs x1#
figx, axx = plt.subplots()
axx.scatter(data['x2'], data['x1'])
xlinfit = xlinreg.params
xfit = xlinfit['x1'] * data['x1']
axx.plot(xfit, data['x1'], label='linear regresion')
axx.set_xlabel('x1')
axx.set_ylabel('x2')
axx.legend()

#Checking colinearity using VIF#
print('\nVIF values:')
xcols = list(data.columns)
xcols.remove('y')
vif_val = []

for test_var in xcols:

    var_rest = list(xcols)
    var_rest.remove(test_var)

    tform = test_var + '~' + '+'.join(var_rest)
    
    tlinreg = smf.ols(formula=tform, data=data[xcols]).fit()

    tvif = 1 / (1 - tlinreg.rsquared)
    
    vif_val.append(tvif)

    print(test_var + ': ' + str(tvif))


#Regresion y vx x1,x2
yallreg = smf.ols(formula='y ~ x1 + x2', data=data).fit()
print('\n\nRegression result: y vs x1, x2')
print(yallreg.summary())

#Residuals and leverage plots#
f, axarr = plt.subplots(2)

#Checking studentized (normalized) residuals for non-linearity and outliers#
sns.regplot(data['y'], yallreg.resid_pearson, lowess=True, ax=axarr[0], line_kws={'color':'r', 'lw':1})
axarr[0].set_title('Normalized residual plot')
axarr[0].set_xlabel('Fitted values')
axarr[0].set_ylabel('Normalized residuals')

#Statsmodels leverage plot#
f = plot_leverage_resid2(yallreg, ax=axarr[1])


#Regresion y vx x1
yx1reg = smf.ols(formula='y ~ x1', data=data).fit()
print('\n\nRegression result: y vs x1')
print(yx1reg.summary())

#Regresion y vx x2
yx2reg = smf.ols(formula='y ~ x2', data=data).fit()
print('\n\nRegression result: y vs x2')
print(yx2reg.summary())


##Adding new data point##
data_new = data.copy()
last = len(data_new.index)
data_new.loc[last, 'x1'] = 0.1
data_new.loc[last, 'x2'] = 0.8
data_new.loc[last, 'y'] = 6

#Regresion y vx x1,x2
ynallreg = smf.ols(formula='y ~ x1 + x2', data=data_new).fit()
print('\n\nNew Regression result: y vs x1, x2')
print(ynallreg.summary())

#Residuals and leverage plots#
fn, axarrn = plt.subplots(2)

#Checking studentized (normalized) residuals for non-linearity and outliers#
sns.regplot(data_new['y'], ynallreg.resid_pearson, lowess=True, ax=axarrn[0], line_kws={'color':'r', 'lw':1})
axarrn[0].set_title('Normalized residual plot')
axarrn[0].set_xlabel('Fitted values')
axarrn[0].set_ylabel('Normalized residuals')

#Statsmodels leverage plot#
fn = plot_leverage_resid2(ynallreg, ax=axarrn[1])


#Regresion y vx x1
ynx1reg = smf.ols(formula='y ~ x1', data=data_new).fit()
print('\n\nNew Regression result: y vs x1')
print(ynx1reg.summary())

#Regresion y vx x2
ynx2reg = smf.ols(formula='y ~ x2', data=data_new).fit()
print('\n\nNew Regression result: y vs x2')
print(ynx2reg.summary())


plt.show()
