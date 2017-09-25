from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chap6lab1 as lab1

#Simulated Data#
np.random.seed(1)
x = np.random.standard_normal(100)
eps = np.random.standard_normal(100)
y = 1 + 2 * x - 5 * np.power(x, 2) + 3 * np.power(x, 3) + eps

data = pd.DataFrame()
data['y'] = y
data['x'] = x

xcols = ['x']

for i in range(2, 11):

    xcols.append('x' + str(i))
    
    data[xcols[-1]] = np.power(x, i)
    

best_models = lab1.best_subset(data, x=xcols, y='y', nsplits=10)

print('\nLowest CV error best subset model:')
print(best_models.loc[best_models['CVerr'].argmin()])

fwd_models = lab1.forward_sel(data, x=xcols, y='y', nsplits=10)
    
print('\nLowest CV error best forward model:')
print(fwd_models.loc[fwd_models['CVerr'].argmin()])

back_models = lab1.backward_sel(data, x=xcols, y='y', nsplits=10)

print('\nLowest CV error best backward model:')
print(back_models.loc[back_models['CVerr'].argmin()])

fbest, ((axbest1, axbest2), (axbest3, axbest4), (axbest5, axbest6)) = plt.subplots(3, 2, sharex='col')
axbest1.plot(best_models['NumVar'], best_models['AdjR2'])
axbest1.set_ylabel('Adjusted R2')
axbest2.plot(best_models['NumVar'], best_models['AIC'])
axbest2.set_ylabel('AIC')
axbest3.plot(best_models['NumVar'], best_models['Cp'])
axbest3.set_ylabel('Cp')
axbest4.plot(best_models['NumVar'], best_models['BIC'])
axbest4.set_ylabel('BIC')
axbest5.set_xlabel('k')
axbest6.plot(best_models['NumVar'], best_models['CVerr'])
axbest6.set_xlabel('k')
axbest6.set_ylabel('CV error')
fbest.suptitle('Best subset selection')

ffwd, ((axfwd1, axfwd2), (axfwd3, axfwd4), (axfwd5, axfwd6)) = plt.subplots(3, 2, sharex='col')
axfwd1.plot(fwd_models['NumVar'], fwd_models['AdjR2'])
axfwd1.set_ylabel('Adjusted R2')
axfwd2.plot(fwd_models['NumVar'], fwd_models['AIC'])
axfwd2.set_ylabel('AIC')
axfwd3.plot(fwd_models['NumVar'], fwd_models['Cp'])
axfwd3.set_ylabel('Cp')
axfwd4.plot(fwd_models['NumVar'], fwd_models['BIC'])
axfwd4.set_ylabel('BIC')
axfwd5.set_xlabel('k')
axfwd6.plot(fwd_models['NumVar'], fwd_models['CVerr'])
axfwd6.set_xlabel('k')
axfwd6.set_ylabel('CV error')
ffwd.suptitle('Forward subset selection')

fback, ((axback1, axback2), (axback3, axback4), (axback5, axback6)) = plt.subplots(3, 2, sharex='col')
axback1.plot(back_models['NumVar'], back_models['AdjR2'])
axback1.set_ylabel('Adjusted R2')
axback2.plot(back_models['NumVar'], back_models['AIC'])
axback2.set_ylabel('AIC')
axback3.plot(back_models['NumVar'], back_models['Cp'])
axback3.set_ylabel('Cp')
axback4.plot(back_models['NumVar'], back_models['BIC'])
axback4.set_ylabel('BIC')
axback5.set_xlabel('k')
axback6.plot(back_models['NumVar'], back_models['CVerr'])
axback6.set_xlabel('k')
axback6.set_ylabel('CV error')
fback.suptitle('Backward subset selection')

plt.tight_layout
plt.show()
