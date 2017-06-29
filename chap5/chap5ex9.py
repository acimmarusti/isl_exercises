from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
import statsmodels.formula.api as smf

#Load boston dataset from sklearn#
boston = load_boston()

#Columns#
#print(boston['feature_names'])
#Descriptio#
#print(boston['DESCR'])

rawdata = pd.DataFrame(boston.data, columns=boston.feature_names)
rawdata['MEDV'] = boston.target

#Convert to NaN#
data = rawdata.replace(to_replace='None', value=np.nan).copy()

print('\nSample Mean:')
print(data['MEDV'].mean())
print('Sample Mean Std Err:')
print(scipy.stats.sem(data['MEDV']))
print('Sample Median:')
print(data['MEDV'].median())
print('Sample tenth percentile:')
print(np.percentile(data['MEDV'], 10))

#Function for estimating stderrs using bootstrapping#
def bootfn(data, type='mean', repeat=1000):

    boot_table = []

    if type == 'mean':
    
        for ite in range(repeat):
    
            data_boot = data.sample(n=len(data.index), replace=True)
        
            boot_table.append(data_boot.mean())

    elif type == 'median':

        for ite in range(repeat):
    
            data_boot = data.sample(n=len(data.index), replace=True)
        
            boot_table.append(data_boot.median())

    else:

        for ite in range(repeat):
    
            data_boot = data.sample(n=len(data.index), replace=True)
        
            boot_table.append(np.percentile(data_boot, 10))
            
    boot_tab = np.array(boot_table)
            
    results = pd.Series()

    results['estimate'] = boot_tab.mean()
    results['stderr'] = boot_tab.std()
    results['t_stat'] = results['estimate'] / results['stderr']
    results['confint_neg'] = results['estimate'] - 1.96 * results['stderr']
    results['confint_pos'] = results['estimate'] + 1.96 * results['stderr']

    return results

print('\nBootstrapped mean')
print(bootfn(data['MEDV']))

print('\nBootstrapped median')
print(bootfn(data['MEDV'], type='median'))

print('\nBootstrapped tenth percentile')
print(bootfn(data['MEDV'], type='percentile'))
