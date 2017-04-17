from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor

filename = '../Auto.csv'

rawdata = pd.read_csv(filename)

#Convert to NaN#
data = rawdata.replace(to_replace='?', value=np.nan).copy()

#Quantitative and qualitative predictors#
print(data.dtypes)

#Simple linear regression#
slinreg = smf.ols('mpg~horsepower', data=data).fit()

print(slinreg.summary())


