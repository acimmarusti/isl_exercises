from __future__ import print_function, division
import numpy as np
import statsmodels.api as sm

#Data#
np.random.seed(1)
x = np.random.standard_normal(100)
y = 2 * x + np.random.standard_normal(100)

#Simple linear regression y vs x#
slinreg = sm.OLS(y, x).fit()

print(slinreg.summary())

#Simple linear regression x vs y#
slinreg2 = sm.OLS(x, y).fit()

print(slinreg2.summary())

"""
This part is giving trouble. This may be a statmodels bug. Looks like sm (without formulas) does not have access to all the OLSResults methods...
#Confirming t statistic manually for y vs x#
yp = slinreg.fittedvalues()

tstat = np.sqrt(len(x) - 1) * np.dot(x, y) / sqrt(np.dot(x, x) * np.dot(yp, yp) - np.square(np.dot(x, yp)))

print('t statistic: ', tstat)
"""

#Adding an intercept#
xn = np.copy(x)
xn = sm.add_constant(xn)

yn = np.copy(y)
yn = sm.add_constant(yn)

#Simple linear regression y vs x#
nslinreg = sm.OLS(y, xn).fit()

print(nslinreg.summary())

#Simple linear regression x vs y#
nslinreg2 = sm.OLS(x, yn).fit()

print(nslinreg2.summary())
