from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


filename = '../Wage.csv'

#Load raw data to pandas dataframe#
rawdata = pd.read_csv(filename)

#Raw data Dimensions#
print('\nRaw data dimensions:')
print(rawdata.shape)

#Column names#
print(rawdata.columns)

#drop the missing values#
data = rawdata.dropna()

#Data dimensions#
print('\nData dimensions after removing NaN:')
print(data.shape)

print(data.dtypes)

lreg = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4)', data=data).fit()

print(lreg.summary())
