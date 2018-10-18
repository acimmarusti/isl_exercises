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

#Polynomial regression#
lreg = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4)', data=data).fit()

print(lreg.summary())

#Polynomial regression confidence intervals#
testdata = pd.DataFrame()
testdata['age'] = np.arange(data['age'].min(), data['age'].max(), dtype=int)

lreg_pred = lreg.get_prediction(testdata, weights=1)

prediction = lreg_pred.summary_frame(alpha=0.05)

testdata[['lpred','lstderr','lmean_ci_lower','lmean_ci_upper','lobs_ci_lower','lobs_ci_upper']] = prediction

#Plot regression and confidence intervals#
f, ax = plt.subplots()
ax.plot(data['age'], data['wage'], 'o')
ax.plot(testdata['age'], testdata['lpred'], 'g-')

ax.plot(testdata['age'], testdata['lobs_ci_lower'], 'r--')
ax.plot(testdata['age'], testdata['lobs_ci_upper'], 'r--')
ax.plot(testdata['age'], testdata['lmean_ci_lower'], 'b--')
ax.plot(testdata['age'], testdata['lmean_ci_upper'], 'b--')

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

#Logistic regression#
data['wage250'] = data['wage'] > 250

greg = smf.glm(formula='wage250~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4)', data=data, family=sm.families.Binomial()).fit()

#Logistic regression confidence intervals#
greg_pred = greg.get_prediction(testdata, weights=1)

gpred = greg_pred.summary_frame(alpha=0.05)

testdata[['gpred','gstderr','gmean_ci_lower','gmean_ci_upper','gobs_ci_lower','gobs_ci_upper']] = np.exp(gpred) / (1 + np.exp(gpred))

#Plot regression and confidence intervals#
fg, axg = plt.subplots()
axg.plot(data['age'], data['wage250'], 'o')
axg.plot(testdata['age'], testdata['gpred'], 'g-')

axg.plot(testdata['age'], testdata['gobs_ci_lower'], 'r--')
axg.plot(testdata['age'], testdata['gobs_ci_upper'], 'r--')
axg.plot(testdata['age'], testdata['gmean_ci_lower'], 'b--')
axg.plot(testdata['age'], testdata['gmean_ci_upper'], 'b--')

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
