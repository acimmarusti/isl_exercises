from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

filename = 'Auto.csv'

rawdata = pd.read_csv(filename)

#Convert to NaN#
data = rawdata.replace(to_replace='?', value=np.nan).copy()

#Quantitative and qualitative predictors#
print(data.dtypes)

#Summary (mean, stdev, range, etc)#
print('Full data summary')
print(data.describe())

#Summary (mean, stdev, range, etc) sliced data#
print('Truncated data summary')
range2drop = range(10,86)
print(data.drop(range2drop).describe())

fig_scatter = scatter_matrix(data)
plt.show()
