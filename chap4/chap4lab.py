from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from pandas.plotting import scatter_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm

filename = '../Smarket.csv'

data = pd.read_csv(filename)

#Convert to NaN#
#data = rawdata.replace(to_replace='?', value=np.nan).copy()
#Convert to NaN#
#data = rawdata.replace(to_replace='None', value=np.nan).copy()

#All columns#
print('\nPredictors')
print(data.columns)

#All predictors#
allpred = list(data.columns)
allpred.remove('Direction')

#Dimensions of dataframe#
print('\nDimensions of data')
print(data.shape)

#Summary (mean, stdev, range, etc)#
print('\nFull data summary')
print(data.describe())

#Correlations#
print('\nData correlations')
print(data.corr())

#Pair plot matrix#
fig_scatter = scatter_matrix(data)

#plot Volumen vs Year#
figv, axv = plt.subplots()
axv.scatter(data['Year'], data['Volume'])
axv.set_xlabel('Year')
axv.set_ylabel('Volume')
axv.legend()

#Logistic regression with statsmodels#
pred_noyr = list(allpred)
pred_noyr.remove('Year')

lr_form = 'Direction~' + '+'.join(pred_noyr)
logreg = smf.glm(formula=lr_form, data=data, family=sm.families.Binomial()).fit()

print('\Logistic regression fit summary')
print(logreg.summary())

plt.show()



