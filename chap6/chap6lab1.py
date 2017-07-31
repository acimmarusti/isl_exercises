from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import statsmodels.formula.api as smf
import statsmodels.api as sm

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

def processSubset(data, x=feats, y=output):

    X = np.array(data[feats])
    Y = np.array(data[output])
    
    # Initiate logistic regression object
    lreg = LinearRegression()

    # Fit model. Let X = matrix of predictors, Y = matrix of variables.
    lreg_res = logit_clf.fit(X, Y)

    #Predicted values for training set
    Y_pred = lreg_res.predict(X)

    #Residual sum of squares#
    rss = np.sum(np.square(Y_pred - Y))

    #total sum of squares#
    tss = np.sum(np.square(Y - np.mean(Y)))

    #Variance estimate#
    var_est = np.square(np.std(Y))

    #Number of predictors#
    kpred = float(len(feats))

    #Number of data points#
    ndat = float(len(Y))
    
    #Cp#
    cp = (rss + 2 * kpred * var_est) / ndat

    #AIC#
    aic = (rss + 2 * kpred * var_est) / (ndat * var_est)

    #BIC#
    bic = (rss + np.log(ndat) * kpred * var_est) / ndat

    #Adjusted R^2#
    ar2 = 1 - (rss / (ndat - kpred - 1)) / (tss / (ndat - 1))

    return {""}
