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
from statsmodels.stats.outliers_influence import variance_inflation_factor, summary_table

filename = '../Auto.csv'

data = pd.read_csv(filename, na_values='?').dropna()

#Quantitative and qualitative predictors#
print(data.dtypes)

#Simple linear regression#
slinreg = smf.ols('mpg ~ horsepower', data=data).fit()

print(slinreg.summary())

st, fitdat, ss2 = summary_table(slinreg, alpha=0.05)

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
plt.show()
