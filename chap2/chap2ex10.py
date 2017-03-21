from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from pandas.tools.plotting import scatter_matrix

boston = load_boston()

#Columns#
print(boston['feature_names'])
#Descriptio#
print(boston['DESCR'])

rawdata = pd.DataFrame(boston.data, columns=boston.feature_names)
rawdata['MEDV'] = boston.target

#Convert to NaN#
data = rawdata.replace(to_replace='None', value=np.nan).copy()

#Are there missing values?#
print('\n\nMissing values?')
print(pd.isnull(data).any())

#Summary (mean, stdev, range, etc)#
print('\n\nFull data summary')
data_sum = data.describe()
print(data_sum)

#Outliers per category#
check_lst = ['CRIM', 'TAX', 'PTRATIO']

mult = 1.0

outlier_thres = np.abs(data_sum.loc['mean']) + mult * data_sum.loc['std']

for pred in check_lst:

    data_out = data[data[pred] >= outlier_thres[pred]]
    if not data_out.empty:
        total_out = len(data_out.index)
        print('\n\nHigh ' + pred + ' rate outliers')
        print('Total : ' + str(total_out))
        print('Correlations to other predictors : ')
        print(data_out.corr(method='pearson')[pred])
    else:
        print('\n\nNo high rate outliers for ' + pred)

#Correlations#
print('\n\nFull data Correlations')
print(data.corr(method='pearson'))

#Bound by river#
print('\n\nSuburbs bound by river')
print(len(data[data['CHAS'] == 1].index))

#Median pupil-teacher ratio#
print('\n\nMedian pupil-teacher ratio')
print(data['PTRATIO'].median())

#Suburb with lowest median value#
print('\n\nLowest median value')
medv_min = data['MEDV'].min()
print(data[data['MEDV'] == medv_min])

#Suburbs with rooms > 7 and 8#
print('\n\nSuburbs with > 7 rooms : ' + str(len(data[data['RM'] > 7].index)))

data_rm8 = data[data['RM'] > 8]
print('\n\nSuburbs with > 8 rooms : ' + str(len(data_rm8.index)))
print('Summary of suburbs > 8 rooms :')
print(data_rm8.describe())

fig_scatter = scatter_matrix(data)
plt.show()
