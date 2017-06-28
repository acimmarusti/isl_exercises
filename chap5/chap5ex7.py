from __future__ import print_function, division
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

filename = '../Weekly.csv'

data = pd.read_csv(filename)

#All predictors#
allpred = list(data.columns)
allpred.remove('Direction')

print(allpred)

#List of predictors#
predictors = ['Lag1', 'Lag2']

## Logistic regression with statsmodels ##
lr_form = 'Direction~' + '+'.join(predictors)

print('\n\n#Full logistic regression#')
logreg = smf.glm(formula=lr_form, data=data, family=sm.families.Binomial()).fit()

print('\n\n#logistic regression without first datum#')
logreg1 = smf.glm(formula=lr_form, data=data[1:], family=sm.families.Binomial()).fit()

print(logreg1.predict(np.transpose(data.loc[1, predictors])))

#print('\nLogistic regression fit summary')
#print(logreg.summary())
