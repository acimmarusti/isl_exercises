from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf
#from sklearn.linear_model import LinearRegression
#import scipy, scipy.stats
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import summary_table

filename = '../Auto.csv'

data = pd.read_csv(filename, na_values='?').dropna()

#Numeric columns#
numcols = list(data.columns)
numcols.remove('name')

#Numeric cols as independent vars#
xcols = list(numcols)
xcols.remove('mpg')

#scatter matrix plot#
fig_scatter = scatter_matrix(data[numcols])

print(data[numcols].corr())

#statsmodels formula#
sform = 'mpg~' + '+'.join(xcols)

#Multi-linear regression#
mlinreg = smf.ols(formula=sform, data=data[numcols]).fit()

print(mlinreg.summary())

#Checking colinearity using VIF#
print('\nVIF values:')
vif_val = []

for test_var in xcols:

    var_rest = list(xcols)
    var_rest.remove(test_var)

    tform = test_var + '~' + '+'.join(var_rest)

    tlinreg = smf.ols(formula=tform, data=data[xcols]).fit()

    tvif = 1 / (1 - tlinreg.rsquared)
    
    vif_val.append(tvif)

    print('\n' + test_var + ': ' + str(tvif))

"""
st, fitdat, ss2 = summary_table(mlinreg, alpha=0.05)

fittedvalues = fitdat[:,2]
predict_mean_se  = fitdat[:,3]
predict_mean_ci_low, predict_mean_ci_upp = fitdat[:,4:6].T
predict_ci_low, predict_ci_upp = fitdat[:,6:8].T

x = data['horsepower']
y = data['mpg']

#Residuals#
resd1 = y - fittedvalues

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.plot(x, y, 'o')
ax1.plot(x, fittedvalues, 'g-')
ax1.plot(x, predict_ci_low, 'r--')
ax1.plot(x, predict_ci_upp, 'r--')
ax1.plot(x, predict_mean_ci_low, 'b--')
ax1.plot(x, predict_mean_ci_upp, 'b--')
ax2.plot(resd1, fittedvalues, 'o')
"""
plt.show()
