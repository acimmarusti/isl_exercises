from __future__ import print_function, division
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

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
print(logreg.summary())

logreg1 = smf.glm(formula=lr_form, data=data[1:], family=sm.families.Binomial()).fit()

#Construct evaluation dataframe for statsmodels#
eval_data = pd.DataFrame({'intercept': 1, 'Lag1': data.loc[0, 'Lag1'], 'Lag2': data.loc[0, 'Lag2']}, index=[0])

print('\n\n#Probability of market going DOWN on first datum#')
print(logreg1.predict(eval_data)[0])
print('#Observed / Real market direction#')
print(data.loc[0, 'Direction'])

proba = []
dir_pred = []
errs = []

for i in range(0, len(list(data.index))):

    #Remove row/data point#
    loo_data = data.drop(data.index[i])

    #Fit logistic regression#
    logreg_ite = smf.glm(formula=lr_form, data=loo_data, family=sm.families.Binomial()).fit()

    #Prepare data for evaluation#
    eval_loo = pd.DataFrame({'intercept': 1, 'Lag1': data.loc[i, 'Lag1'], 'Lag2': data.loc[i, 'Lag2']}, index=[0])

    #Predict#
    proba_pred = logreg_ite.predict(eval_loo)[0]
    proba.append(proba_pred)
        

#Predictions#
pred_mask = np.array(proba) < 0.5
dir_pred = [1 if item else 0 for item in pred_mask]

#Encoding directions to binary array#
dir_real = [1 if item=='Up' else 0 for item in data['Direction']]

#Accuracy calculation#
print('\nAccuracy logit:', np.round(accuracy_score(dir_pred, dir_real), 3))
