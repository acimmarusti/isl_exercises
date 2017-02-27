from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

filename = 'College.csv'

rawdata = pd.read_csv(filename)

fixdata = rawdata.rename(columns={rawdata.columns[0]: 'college'})

print(fixdata.columns)

fixdata['Elite'] = 'No'
theelite = fixdata[fixdata['Top10perc'] > 50].index
fixdata.at[theelite, 'Elite'] = 'Yes'

num_elite = len(fixdata[fixdata['Elite'] == 'Yes'].index)

print('Number of Elite institutions: ', num_elite)

data_summary = fixdata.describe()

print(data_summary)

fig_scatter = scatter_matrix(fixdata.iloc[:,1:10])
plt.show()

fig_priv = sns.boxplot(x='Private', y='Outstate', data=fixdata)
sns.plt.show()
fig_elite = sns.boxplot(x='Elite', y='Outstate', data=fixdata)
sns.plt.show()
