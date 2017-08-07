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


def processSubset(data, x=[], y=[]):

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

    return {"Vars": ";".join(x), "model": lreg_res, "NumVar": kpred, "RSE": rse, "R2": r2, "Cp": cp, "AIC": aic, "BIC": bic, "AdjR2": ar2}

kvars = 2

tic = time.time()

results = []

for combo in itertools.combinations(xcols, kvars):
    results.append(processSubset(data, x=list(combo), y='Salary'))

#Wrap everything in DataFrame#
allmodels = pd.DataFrame(results)

best_kmodel = allmodels.loc[allmodels['R2'].argmax()]

toc = time.time()

print(best_kmodel)
print(toc-tic)



