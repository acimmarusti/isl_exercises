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
