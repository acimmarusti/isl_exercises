from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

boston = load_boston()

#Columns#
print(boston['feature_names'])
#Descriptio#
print(boston['DESCR'])

rawdata = pd.DataFrame(boston.data, columns=boston.feature_names)
rawdata['MEDV'] = boston.target

#Convert to NaN#
data = rawdata.replace(to_replace='None', value=np.nan).copy()

#Simple linear regression#
slinreg = smf.ols('MEDV~LSTAT', data=data).fit()

print(slinreg.summary())

#Multi-linear regression#
mlinreg = smf.ols('MEDV~LSTAT+AGE', data=data).fit()

print(mlinreg.summary())

#All-linear regression#
allcols = list(data.columns)
allcols.remove('MEDV')
allcols = '+'.join(allcols)
alinreg = smf.ols('MEDV~' + allcols, data=data).fit()

print(alinreg.summary())

#All-but linear regression#
ablinreg = smf.ols('MEDV~' + allcols + '-AGE', data=data).fit()

print(ablinreg.summary())

#Interaction linear regression#
ilinreg = smf.ols('MEDV~LSTAT*AGE', data=data).fit()

print(ilinreg.summary())

#Non-linear terms linear regression#
nlinreg = smf.ols('MEDV~LSTAT+np.square(AGE)', data=data).fit()

print(nlinreg.summary())

plt.figure()
plt.plot(data['LSTAT'], data['MEDV'], 'o')
plt.plot(data['LSTAT'], slinreg.fittedvalues, 'r--')
plt.show()
