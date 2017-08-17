from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import time
import itertools
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

filename = '../Hitters.csv'

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

#Only numerical cols#
colmask = data.dtypes != object
numcols = colmask.index[colmask == True]
xcols = list(numcols)
xcols.remove('Salary')


def processSubset(data, x=['x'], y=['y']):

    X = np.array(data[x])
    Y = np.array(data[y])

    # Initiate logistic regression object
    lreg = LinearRegression()

    # Fit model. Let X = matrix of predictors, Y = matrix of variables.
    lreg_res = lreg.fit(X, Y)

    #Predicted values for training set
    Y_pred = lreg_res.predict(X)

    #Number of predictors#
    kpred = float(len(x))

    #Number of data points#
    ndat = float(len(Y))

    #Residual sum of squares#
    rss = np.sum(np.square(Y_pred - Y))

    #total sum of squares#
    tss = np.sum(np.square(Y - np.mean(Y)))

    #R squared#
    r2 = 1 - rss / tss

    #Residual standard error#
    rse = np.sqrt(rss / (ndat - kpred - 1))

    #Variance estimate#
    var_est = np.square(np.std(Y))

    #Cp#
    cp = (rss + 2 * kpred * var_est) / ndat

    #AIC#
    aic = (rss + 2 * kpred * var_est) / (ndat * var_est)

    #BIC#
    bic = (rss + np.log(ndat) * kpred * var_est) / ndat

    #Adjusted R^2#
    ar2 = 1 - (rss / (ndat - kpred - 1)) / (tss / (ndat - 1))

    return {"Vars": ";".join(x), "model": lreg_res, "NumVar": kpred, "RSS": rss, "RSE": rse, "R2": r2, "Cp": cp, "AIC": aic, "BIC": bic, "AdjR2": ar2}


def getBest(data, x=['x'], y=['y'], k=2):

    tic = time.time()

    results = []

    for combo in itertools.combinations(x, k):
        results.append(processSubset(data, x=list(combo), y=y))

    #Wrap everything in DataFrame#
    allmodels = pd.DataFrame(results)

    #Choose the model with lowest RSS
    best_kmodel = allmodels.loc[allmodels['RSS'].argmin()]

    toc = time.time()

    print("Processed ", len(allmodels.index), "models on", k, "predictors in", toc-tic, "seconds")

    return best_kmodel


def forward(data, preds, x=['x'], y=['y']):

    #Get only predictors that still need to be processed#
    remain = [p for p in x if p not in preds]

    tic = time.time()

    results = []

    for p in remain:
        results.append(processSubset(data, x=preds + [p], y=y))

    #Wrap everything in DataFrame#
    allmodels = pd.DataFrame(results)

    #Choose the model with lowest RSS
    best_model = allmodels.loc[allmodels['RSS'].argmin()]

    toc = time.time()

    print("Processed ", len(allmodels.index), "models on", len(preds) + 1, "predictors in", toc-tic, "seconds")

    return best_model


def backward(data, preds, x=['x'], y=['y']):

    tic = time.time()

    results = []

    for combo in itertools.combinations(preds, len(preds) - 1):
        results.append(processSubset(data, x=list(combo), y=y))

    #Wrap everything in DataFrame#
    allmodels = pd.DataFrame(results)

    #Choose the model with lowest RSS
    best_model = allmodels.loc[allmodels['RSS'].argmin()]

    toc = time.time()

    print("Processed ", len(allmodels.index), "models on", len(preds) + 1, "predictors in", toc-tic, "seconds")

    return best_model


def best_subset(data, x=['x'], y=['y']):

    models = pd.DataFrame(columns=['Vars','model','NumVar','RSS','RSE','R2','Cp','AIC','BIC','AdjR2'])

    tic = time.time()

    for ii in range(1, len(x) + 1):

        models.loc[ii] = getBest(data, x=x, y=y, k=ii)

    toc = time.time()
    print("Total elapsed time:", toc-tic, "seconds")

    return models


def forward_sel(data, x=['x'], y=['y']):

    models = pd.DataFrame(columns=['Vars','model','NumVar','RSS','RSE','R2','Cp','AIC','BIC','AdjR2'])

    tic = time.time()

    pred = []

    for ii in range(1, len(x) + 1):

        models.loc[ii] = forward(data, pred, x=x, y=y)
        pred = models.at[ii, 'Vars'].split(';')

    toc = time.time()
    print("Total elapsed time:", toc-tic, "seconds")

    return models


def backward_sel(data, x=['x'], y=['y']):

    models = pd.DataFrame(columns=['Vars','model','NumVar','RSS','RSE','R2','Cp','AIC','BIC','AdjR2'])

    tic = time.time()

    pred = xcols

    while(len(pred) > 1):

        models.loc[len(pred) - 1] = backward(data, pred, x=x, y=y)
        pred = models.at[len(pred) - 1, 'Vars'].split(';')

    toc = time.time()
    print("Total elapsed time:", toc-tic, "seconds")

    return models


best_models = best_subset(data, x=xcols, y='Salary')

fwd_models = forward_sel(data, x=xcols, y='Salary')

back_models = backward_sel(data, x=xcols, y='Salary')


fbest, ((axbest1, axbest2), (axbest3, axbest4)) = plt.subplots(2, 2, sharex='col')
axbest1.plot(best_models['NumVar'], best_models['AdjR2'])
axbest1.set_ylabel('Adjusted R2')
axbest2.plot(best_models['NumVar'], best_models['AIC'])
axbest2.set_ylabel('AIC')
axbest3.plot(best_models['NumVar'], best_models['Cp'])
axbest3.set_xlabel('k')
axbest3.set_ylabel('Cp')
axbest4.plot(best_models['NumVar'], best_models['BIC'])
axbest4.set_xlabel('k')
axbest4.set_ylabel('BIC')
fbest.suptitle('Best subset selection')

ffwd, ((axfwd1, axfwd2), (axfwd3, axfwd4)) = plt.subplots(2, 2, sharex='col')
axfwd1.plot(fwd_models['NumVar'], fwd_models['AdjR2'])
axfwd1.set_ylabel('Adjusted R2')
axfwd2.plot(fwd_models['NumVar'], fwd_models['AIC'])
axfwd2.set_ylabel('AIC')
axfwd3.plot(fwd_models['NumVar'], fwd_models['Cp'])
axfwd3.set_xlabel('k')
axfwd3.set_ylabel('Cp')
axfwd4.plot(fwd_models['NumVar'], fwd_models['BIC'])
axfwd4.set_xlabel('k')
axfwd4.set_ylabel('BIC')
ffwd.suptitle('Forward subset selection')

fback, ((axback1, axback2), (axback3, axback4)) = plt.subplots(2, 2, sharex='col')
axback1.plot(back_models['NumVar'], back_models['AdjR2'])
axback1.set_ylabel('Adjusted R2')
axback2.plot(back_models['NumVar'], back_models['AIC'])
axback2.set_ylabel('AIC')
axback3.plot(back_models['NumVar'], back_models['Cp'])
axback3.set_xlabel('k')
axback3.set_ylabel('Cp')
axback4.plot(back_models['NumVar'], back_models['BIC'])
axback4.set_xlabel('k')
axback4.set_ylabel('BIC')
fback.suptitle('Backward subset selection')

plt.tight_layout
plt.show()
