from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_leverage_resid2
from statsmodels.stats.outliers_influence import summary_table
#from sklearn.linear_model import LinearRegression
#import scipy, scipy.stats
#from statsmodels.sandbox.regression.predstd import wls_prediction_std

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

f, axarr = plt.subplots(2)
#Checking studentized (normalized) residuals for non-linearity and outliers#
sns.regplot(data['mpg'], mlinreg.resid_pearson, lowess=True, ax=axarr[0], line_kws={'color':'r', 'lw':1})
axarr[0].set_title('Normalized residual plot')
axarr[0].set_xlabel('Fitted values')
axarr[0].set_ylabel('Normalized residuals')

#Statsmodels leverage plot#
f = plot_leverage_resid2(mlinreg, ax=axarr[1])

#Checking interaction terms#
inter_test = smf.ols(formula='mpg~ cylinders*displacement + cylinders*horsepower + cylinders*weight + displacement*horsepower + displacement*weight + horsepower*weight', data=data[numcols]).fit()

print(inter_test.summary())

#Final "best" fit#
final_fit = smf.ols(formula='np.log(mpg) ~ horsepower + weight + origin + year + np.square(horsepower) + np.square(weight) + np.square(year)', data=data[numcols]).fit()

print(final_fit.summary())

fig, axf = plt.subplots()
#Checking studentized (normalized) residuals for non-linearity and outliers#
sns.regplot(data['mpg'], final_fit.resid_pearson, lowess=True, ax=axf, line_kws={'color':'r', 'lw':1})
axf.set_title('Normalized residual plot')
axf.set_xlabel('Fitted values')
axf.set_ylabel('Normalized residuals')


plt.show()
