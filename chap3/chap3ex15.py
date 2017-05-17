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

allparams = []

for var in allcols:

    slinreg = smf.ols('MEDV~' + var + '-1', data=data).fit()
    allparams.append(slinreg.params)
    
#All-linear regression#
col_str = '+'.join(allcols)
alinreg = smf.ols('MEDV~' + col_str, data=data).fit()

print(alinreg.summary())


#plt.show()
