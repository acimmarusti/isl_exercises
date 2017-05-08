from __future__ import print_function, division
import numpy as np
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
#from statsmodels.graphics.regressionplots import abline_plot

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

#Regressions x2 vs x1 (removing constant or intercept term)#
xlinreg = smf.ols(formula='x2 ~ x1 - 1', data=data).fit()
print(xlinreg.summary())

#plots x2 vs x1#
figx, axx = plt.subplots()
axx.scatter(data['x2'], data['x1'])
xlinfit = xlinreg.params
xfit = xlinfit['x1'] * data['x1']
axx.plot(xfit, data['x1'], label='linear regresion')
axx.set_xlabel('x1')
axx.set_ylabel('x2')
axx.legend()

"""
#Regression y2 vs x#
slin2reg = smf.ols(formula='y2 ~ x', data=data).fit()
print(slin2reg.summary())

#plots y2#
fig2, axf2 = plt.subplots(2, sharex=True)
axf2[0].scatter(data['y2'], data['x'])
lin2fit = slin2reg.params
y2fit = lin2fit['Intercept'] + lin2fit['x'] * data['x']
axf2[0].plot(y2fit, data['x'], label='linear regresion')
axf2[0].set_xlabel('x')
axf2[0].set_ylabel('y')
sns.regplot(y=slin2reg.resid_pearson, x=data['x'], lowess=True, ax=axf2[1], line_kws={'color':'r', 'lw':1})
axf2[1].set_xlabel('x')
axf2[1].set_ylabel('Residuals poly regression')


#Regression y3 vs x#
slin3reg = smf.ols(formula='y3 ~ x', data=data).fit()
print(slin3reg.summary())

#plots y3#
fig3, axf3 = plt.subplots(2, sharex=True)
axf3[0].scatter(data['y3'], data['x'])
lin3fit = slin3reg.params
y3fit = lin3fit['Intercept'] + lin3fit['x'] * data['x']
axf3[0].plot(y2fit, data['x'], label='linear regresion')
axf3[0].set_xlabel('x')
axf3[0].set_ylabel('y')
sns.regplot(y=slin3reg.resid_pearson, x=data['x'], lowess=True, ax=axf3[1], line_kws={'color':'r', 'lw':1})
axf3[1].set_xlabel('x')
axf3[1].set_ylabel('Residuals poly regression')
"""
plt.show()
