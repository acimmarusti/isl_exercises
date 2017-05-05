from __future__ import print_function, division
import numpy as np
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import abline_plot

#Generate data#
np.random.seed(1)
x = np.random.standard_normal(100)
eps = np.random.normal(loc=0, scale=0.25, size=100)
eps2 = np.random.normal(loc=0, scale=0.1, size=100)
eps3 = np.random.normal(loc=0, scale=0.5, size=100)

y1 = -1 + 0.5 * x + eps
y2 = -1 + 0.5 * x + eps2
y3 = -1 + 0.5 * x + eps3

#Pandas dataframe#
data = pd.DataFrame()
data['x'] = x
data['y1'] = y1
data['y2'] = y2
data['y3'] = y3

#Regressions y1 vs x#
slinreg = smf.ols(formula='y1 ~ x', data=data).fit()
print(slinreg.summary())

slinreg2 = smf.ols(formula='y1 ~ x + np.square(x)', data=data).fit()
print(slinreg2.summary())

#plots y1#
fig, axf = plt.subplots(2, sharex=True)
axf[0].scatter(data['y1'], data['x'])
abline_plot(model_results=slinreg, ax=axf[0])
abline_plot(model_results=slinreg2, ax=axf[0])
#axf[0].plot(slinreg.fittedvalues(), data['x'], label='linear regresion')
#axf[0].plot(slinreg2.fittedvalues(), data['x'], label='quadratic regresion')
axf[0].set_ylabel('y')

sns.regplot(y=slinreg2.resid_pearson, x=data['x'], lowess=True, ax=axf[1], line_kws={'color':'r', 'lw':1})
axf[1].set_xlabel('x')
axf[1].set_ylabel('Residuals poly regression')


#Regression y2 vs x#
slin2reg = smf.ols(formula='y2 ~ x', data=data).fit()
print(slin2reg.summary())

#plots y2#
fig2, axf2 = plt.subplots()
axf2.scatter(data['y2'], data['x'])
axf2.plot(slin2reg.fittedvalues(), data['x'], label='linear regresion')
axf2.set_xlabel('x')
axf2.set_ylabel('y')


#Regression y3 vs x#
slin3reg = smf.ols(formula='y3 ~ x', data=data).fit()
print(slin3reg.summary())

#plots y3#
fig3, axf3 = plt.subplots()
axf3.scatter(data['y3'], data['x'])
axf3.plot(slin3reg.fittedvalues(), data['x'], label='linear regresion')
axf3.set_xlabel('x')
axf3.set_ylabel('y')

plt.show()
