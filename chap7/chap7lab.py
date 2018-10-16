from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std


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

testdata = pd.DataFrame()
testdata['age'] = np.arange(data['age'].min(), data['age'].max(), dtype=int)

lreg_pred = lreg.get_prediction(testdata, weights=1)

prediction = lreg_pred.summary_frame(alpha=0.05)

testdata[['pred','stderr','mean_ci_lower','mean_ci_upper','obs_ci_lower','obs_ci_upper']] = prediction

f, ax = plt.subplots()
ax.plot(data['age'], data['wage'], 'o')
ax.plot(testdata['age'], testdata['pred'], 'g-')

ax.plot(testdata['age'], testdata['obs_ci_lower'], 'r--')
ax.plot(testdata['age'], testdata['obs_ci_upper'], 'r--')
ax.plot(testdata['age'], testdata['mean_ci_lower'], 'b--')
ax.plot(testdata['age'], testdata['mean_ci_upper'], 'b--')

#ANOVA#

lreg1 = smf.ols(formula='wage~age', data=data).fit()
lreg2 = smf.ols(formula='wage~age + np.power(age, 2)', data=data).fit()
lreg3 = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3)', data=data).fit()
lreg4 = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4)', data=data).fit()
lreg5 = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4) + np.power(age, 5)', data=data).fit()

anova_table = sm.stats.anova_lm(lreg1, lreg2, lreg3, lreg4, lreg5)

print('\nANOVA comparing different polynomial fits: wage vs. age')
print(anova_table)

lregew1 = smf.ols(formula='wage~education + age', data=data).fit()
lregew2 = smf.ols(formula='wage~education + age + np.power(age, 2)', data=data).fit()
lregew3 = smf.ols(formula='wage~education + age + np.power(age, 2) + np.power(age, 3)', data=data).fit()

anova_table2 = sm.stats.anova_lm(lregew1, lregew2, lregew3)

print('\nANOVA comparing different polynomial fits: wage vs. eduction and age')
print(anova_table2)

plt.show()
"""
st, fitdata, ss2 = summary_table(lreg, alpha=0.05)

fittedvalues = fitdata[:,2]
predict_mean_se  = fitdata[:,3]
predict_mean_ci_low, predict_mean_ci_upp = fitdata[:,4:6].T
predict_ci_low, predict_ci_upp = fitdata[:,6:8].T

# check we got the right things
print np.max(np.abs(lreg.fittedvalues - fittedvalues))
print np.max(np.abs(iv_l - predict_ci_low))
print np.max(np.abs(iv_u - predict_ci_upp))

plt.plot(data['age'], data['wage'], 'o')
plt.plot(data['age'], fittedvalues, '-', lw=2)
plt.plot(data['age'], predict_ci_low, 'r--', lw=2)
plt.plot(data['age'], predict_ci_upp, 'r--', lw=2)
plt.plot(data['age'], predict_mean_ci_low, 'r--', lw=2)
plt.plot(data['age'], predict_mean_ci_upp, 'r--', lw=2)
plt.show()
"""
