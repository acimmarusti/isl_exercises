from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import interpolate

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

#Define range for predition test data#
testdata = pd.DataFrame()
testdata['age'] = np.arange(data['age'].min(), data['age'].max(), dtype=int)


#Polynomial regression confidence intervals#
lreg_pred = lreg.get_prediction(testdata, weights=1)

prediction = lreg_pred.summary_frame(alpha=0.05)

testdata[['lpred','lstderr','lmean_ci_lower','lmean_ci_upper','lobs_ci_lower','lobs_ci_upper']] = prediction

#Plot regression and confidence intervals#
f, ax = plt.subplots()
ax.scatter(data['age'], data['wage'], facecolor='None', edgecolor='k', alpha=0.1)
ax.plot(testdata['age'], testdata['lpred'], 'g-')

#ax.plot(testdata['age'], testdata['lobs_ci_lower'], 'r--')
#ax.plot(testdata['age'], testdata['lobs_ci_upper'], 'r--')
ax.plot(testdata['age'], testdata['lmean_ci_lower'], 'g--', alpha=0.8)
ax.plot(testdata['age'], testdata['lmean_ci_upper'], 'g--', alpha=0.8)

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
print(greg.summary())

"""FIX: TypeError get_prediction() got an unexpected keyword argument 'weights' 
#Logistic regression confidence intervals#
greg_pred = greg.get_prediction(testdata, weights=1)

gpred = greg_pred.summary_frame(alpha=0.05)

testdata[['gpred','gstderr','gmean_ci_lower','gmean_ci_upper','gobs_ci_lower','gobs_ci_upper']] = np.exp(gpred) / (1 + np.exp(gpred))

#Plot regression and confidence intervals#
fg, axg = plt.subplots()
#axg.scatter(data['age'], data['wage250'], facecolor='None', edgecolor='k', alpha=0.1)
axg.plot(testdata['age'], testdata['gpred'], 'g-')

#axg.plot(testdata['age'], testdata['gobs_ci_lower'], 'r--')
#axg.plot(testdata['age'], testdata['gobs_ci_upper'], 'r--')
axg.plot(testdata['age'], testdata['gmean_ci_lower'], 'g--', alpha=0.8)
axg.plot(testdata['age'], testdata['gmean_ci_upper'], 'g--', alpha=0.8)
"""

#Fit step function#
data['agebin'] = pd.cut(data['age'], 4)

#linear regression#
lbreg = smf.ols(formula='wage~agebin', data=data).fit()

print(lbreg.summary())

#Get intervals from data and apply to testdata#
interval_list = data['agebin'].unique().tolist()
cuts = [interval.left for interval in interval_list]
cuts.append(interval_list[-1].right)
testdata['agebin'] = pd.cut(testdata['age'], cuts)

lbreg_pred = lbreg.get_prediction(testdata, weights=1)

lb_pred = lbreg_pred.summary_frame(alpha=0.05)

testdata[['lbpred','lbstderr','lbmean_ci_lower','lbmean_ci_upper','lbobs_ci_lower','lbobs_ci_upper']] = lb_pred

#Plot regression and confidence intervals#
flb, axlb = plt.subplots()
axlb.scatter(data['age'], data['wage'], facecolor='None', edgecolor='k', alpha=0.1)
axlb.plot(testdata['age'], testdata['lbpred'], 'g-')

#axlb.plot(testdata['age'], testdata['lbobs_ci_lower'], 'r--')
#axlb.plot(testdata['age'], testdata['lbobs_ci_upper'], 'r--')
axlb.plot(testdata['age'], testdata['lbmean_ci_lower'], 'g--', alpha=0.8)
axlb.plot(testdata['age'], testdata['lbmean_ci_upper'], 'g--', alpha=0.8)


##Splines##
tck = interpolate.splrep(data['age'], np.array(data['wage'], s=0)
data['cubic'] = interpolate.splev(data['age'], tck, der=0)
testdata['cubic'] = interpolate.splev(testdata['age'], tck, der=0)

lcbreg = smf.ols(formula='wage~cubic', data=data).fit()

#Polynomial regression confidence intervals#
lcreg_pred = lcreg.get_prediction(testdata, weights=1)

lcpred = lcreg_pred.summary_frame(alpha=0.05)

testdata[['lcpred','lcstderr','lcmean_ci_lower','lcmean_ci_upper','lcobs_ci_lower','lcobs_ci_upper']] = lcpred

#Plot regression and confidence intervals#
fc, axc = plt.subplots()
axc.scatter(data['age'], data['wage'], facecolor='None', edgecolor='k', alpha=0.1)
axc.plot(testdata['age'], testdata['lcpred'], 'g-')

#axc.plot(testdata['age'], testdata['lobs_ci_lower'], 'r--')
#axc.plot(testdata['age'], testdata['lobs_ci_upper'], 'r--')
axc.plot(testdata['age'], testdata['lcmean_ci_lower'], 'g--', alpha=0.8)
axc.plot(testdata['age'], testdata['lcmean_ci_upper'], 'g--', alpha=0.8)


plt.show()
