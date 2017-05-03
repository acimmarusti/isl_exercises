from __future__ import print_function, division
import numpy as np
import statsmodels.api as sm

###With differnt coefficient###
np.random.seed(1)
x = np.random.standard_normal(100)
y = 2 * x + np.random.standard_normal(100)

#Simple linear regression y vs x#
slinreg = sm.OLS(y, x).fit()

print(slinreg.summary())

#Simple linear regression x vs y#
slinreg2 = sm.OLS(x, y).fit()

print(slinreg2.summary())


###With the same coefficient###
xn = np.copy(x)

#yn = -np.random.choice(xn, size=len(xn))
yn = -np.copy(x)
np.random.shuffle(yn)

#condition for same coefficient#
print('Sum of Xi^2: ' + str(np.dot(xn, xn)))
print('Sum of Yi^2: ' + str(np.dot(yn, yn)))

#Simple linear regression y vs x#
nslinreg = sm.OLS(yn, xn).fit()

print(nslinreg.summary())

#Simple linear regression x vs y#
nslinreg2 = sm.OLS(xn, yn).fit()

print(nslinreg2.summary())
