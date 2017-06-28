from __future__ import print_function, division
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

#Get linear fit parameter function#
def get_logreg_param(data, ylabel='y', xlabel='x'):

    bp_form = ylabel + '~' + '+'.join(xlabel)

    return smf.glm(formula=bp_form, data=data, family=sm.families.Binomial()).fit().params


def bootfn(data, target='y', predictor='x', repeat=1000):

    boot_table = pd.DataFrame()
    
    for ite in range(repeat):
    
        data_boot = data.sample(n=len(data.index), replace=True)

        boot_table[str(ite+1)] = get_logreg_param(data_boot, ylabel=target, xlabel=predictor)

    results = pd.DataFrame()

    boot_tab = boot_table.transpose()
    
    results['estimate'] = boot_tab.mean()
    results['stderr'] = boot_tab.std()

    return results

filename = '../Default.xlsx'

#Load data to pandas dataframe and drop the missing values#
data = pd.read_excel(filename)

print('\n\n### LOGISTIC REGRESSION###')

## Logistic regression with statsmodels ##
preds = ['income', 'balance']
lr_form = 'default~' + '+'.join(preds)
logreg = smf.glm(formula=lr_form, data=data, family=sm.families.Binomial()).fit()

print('\nLogistic regression fit summary')
print(logreg.summary())

print('\nLogistic regression stderr using bootstrapping')
print(bootfn(data, target='default', predictor=preds))
