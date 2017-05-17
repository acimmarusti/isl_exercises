from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf

boston = load_boston()

#Columns#
#print(boston['feature_names'])
#Descriptio#
#print(boston['DESCR'])

rawdata = pd.DataFrame(boston.data, columns=boston.feature_names)
rawdata['MEDV'] = boston.target

#Convert to NaN#
data = rawdata.replace(to_replace='None', value=np.nan).copy()

#Non-response columns#
allcols = list(data.columns)
allcols.remove('MEDV')

allparams = pd.Series(index=allcols)

for var in allcols:

    slinreg = smf.ols('MEDV~' + var, data=data).fit()
    allparams[var] = slinreg.params[var]
    if slinreg.pvalues[var] > 0.01:
        print(var + ' fit coefficient has p-value larger than 1%')
        print(var + ' fit has R2 = ' + str(slinreg.rsquared))
        print(var + ' fit has F-statistic = ' + str(slinreg.fvalue) + ' with p-value of ' + str(slinreg.f_pvalue))
   

#All-linear regression#
col_str = '+'.join(allcols)
alinreg = smf.ols('MEDV~' + col_str, data=data).fit()

print(alinreg.summary())

#plot univariate reg coeff vs multi-reg coeff#
fig, ax = plt.subplots()
ax.scatter(alinreg.params[allcols], allparams)
ax.set_xlabel('univariate coeff')
ax.set_ylabel('multiple reg coeff')

plt.show()
