from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf

boston = load_boston()

#Columns#
#print(boston['feature_names'])
#Descriptio#
#print(boston['DESCR'])

rawdata = pd.DataFrame(boston.data, columns=boston.feature_names)
rawdata['MEDV'] = boston.target

#Convert to NaN#
data = rawdata.replace(to_replace='None', value=np.nan).copy()

#Non-response columns#
allcols = list(data.columns)
allcols.remove('CRIM')

#Linear regression for each predictor#
lin_par = pd.DataFrame(columns=['B1', 'B1p', 'R2', 'Fval', 'Fpval'])

for var in allcols:

    slinreg = smf.ols('CRIM~' + var, data=data).fit()

    #Saving important fit metrics#
    lin_par.loc[var, 'B1'] = slinreg.params[var]
    lin_par.loc[var, 'B1p'] = slinreg.pvalues[var]
    lin_par.loc[var, 'R2'] = slinreg.rsquared
    lin_par.loc[var, 'Fval'] = slinreg.fvalue
    lin_par.loc[var, 'Fpval'] = slinreg.f_pvalue

    if slinreg.pvalues[var] > 0.01:
        print('\n')
        print(var + ' fit coefficient has p-value = ' + str(slinreg.pvalues[var]))
        print(var + ' fit has R2 = ' + str(slinreg.rsquared))
        print(var + ' fit has F-statistic = ' + str(slinreg.fvalue) + ' with p-value = ' + str(slinreg.f_pvalue))
        print('\n')
   

#Multiple linear regression#
col_str = '+'.join(allcols)
alinreg = smf.ols('CRIM~' + col_str, data=data).fit()
print(alinreg.summary())

#Parameters that reject null hypothesis#
nnull_par = alinreg.pvalues[alinreg.pvalues < 0.04]
print('\nParameters that reject null hypothesis')
print(list(nnull_par.index))

#plot univariate reg coeff vs multi-reg coeff#
fig, ax = plt.subplots()
ax.scatter(lin_par['B1'], alinreg.params[allcols])
ax.set_xlabel('univariate coeff')
ax.set_ylabel('multiple reg coeff')

#Non-Linear regression for each predictor#
nlin_par = pd.DataFrame(columns=['R2', 'Fval', 'Fpval'])

for var in allcols:

    mform = 'CRIM ~ ' + var + ' + np.power(' + var + ', 2)' + '+ np.power(' + var + ', 3)'
    
    mlinreg = smf.ols(formula=mform, data=data).fit()

    #Saving important fit metrics#
    nlin_par.loc[var, 'R2'] = mlinreg.rsquared
    nlin_par.loc[var, 'Fval'] = mlinreg.fvalue
    nlin_par.loc[var, 'Fpval'] = mlinreg.f_pvalue

    #Show which parameters have high p-values#
    hi_pval = mlinreg.pvalues[mlinreg.pvalues > 0.04]
    lo_pval = mlinreg.pvalues[mlinreg.pvalues < 0.04]

    print('\n')
    if not lo_pval.empty:
        print(var + ' fit null hypothesis can be rejected for:')
        print(list(lo_pval.index))
        
    if not hi_pval.empty:

        for par in hi_pval.index:
            print(var + ' fit coefficient ' + par + ' has p-value = ' + str(mlinreg.pvalues[par]))
        print(var + ' fit has R2 = ' + str(mlinreg.rsquared))
        print(var + ' fit has F-statistic = ' + str(mlinreg.fvalue) + ' with p-value = ' + str(mlinreg.f_pvalue))

    print('\n')
        
plt.show()
