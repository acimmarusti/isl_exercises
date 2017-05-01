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

filename = '../Carseats.csv'

data = pd.read_csv(filename, na_values='?', index_col=0).dropna()

#Multi-linear regression#
mlinreg = smf.ols(formula='Sales ~ Price + Urban + US', data=data).fit()

print(mlinreg.summary())

#Multi-linear regression without Urban#
mlinreg2 = smf.ols(formula='Sales ~ Price + US', data=data).fit()

print(mlinreg2.summary())

f, axarr = plt.subplots(2)
#Checking studentized (normalized) residuals for non-linearity and outliers#
sns.regplot(data['Sales'], mlinreg2.resid_pearson, lowess=True, ax=axarr[0], line_kws={'color':'r', 'lw':1})
axarr[0].set_title('Normalized residual plot')
axarr[0].set_xlabel('Fitted values')
axarr[0].set_ylabel('Normalized residuals')

#Statsmodels leverage plot#
f = plot_leverage_resid2(mlinreg2, ax=axarr[1])

plt.show()
