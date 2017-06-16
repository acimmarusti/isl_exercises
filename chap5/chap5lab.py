from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm

filename = '../Auto.csv'

#Load data to pandas dataframe and drop the missing values#
data = pd.read_csv(filename, na_values='?').dropna()

#Add non-linear terms#
data['horsepower2'] = np.power(data['horsepower'], 2)
data['horsepower3'] = np.power(data['horsepower'], 3)

#Random sampling#
data_train = data.sample(n=196, random_state=2)
data.loc[data.index, 'train'] = 'n'
data.loc[data_train.index, 'train'] = 'y' 
data_test = data[data['train'] == 'n']

#Numeric columns#
numcols = list(data.columns)
numcols.remove('name')
numcols.remove('mpg')
numcols.remove('train')


print('\n\n### LINEAR REGRESSION WITH STATSMODELS###')

## Linear regression with statsmodels ##
lreg = smf.ols(formula='mpg~horsepower', data=data_train).fit()
print(np.mean(np.power(data_test['mpg'] - lreg.predict(data_test['horsepower']), 2)))

l2reg = smf.ols(formula='mpg~horsepower + np.power(horsepower, 2)', data=data_train).fit()
print(np.mean(np.power(data_test['mpg'] - l2reg.predict(data_test['horsepower']), 2)))

l3reg = smf.ols(formula='mpg~horsepower + np.power(horsepower, 2) + np.power(horsepower, 3)', data=data_train).fit()
print(np.mean(np.power(data_test['mpg'] - l3reg.predict(data_test['horsepower']), 2)))

#print(lreg.summary())
#print(l2reg.summary())
#print(l3reg.summary())

print('\n\n### LINEAR REGRESSION WITH SKLEARN###')

#Reshaping data into sklearn's preferred format#
train_size = len(data_train.index)
y_train = np.reshape(data_train['mpg'], (train_size, 1))
x_train = np.reshape(data_train['horsepower'], (train_size, 1))
x2_train = np.reshape(data_train[['horsepower', 'horsepower2']], (train_size, 2))
x3_train = np.reshape(data_train[['horsepower', 'horsepower2', 'horsepower3']], (train_size, 3))

test_size = len(data_test.index)
y_test = np.reshape(data_test['mpg'], (test_size, 1))
x_test = np.reshape(data_test['horsepower'], (test_size, 1))
x2_test = np.reshape(data_test[['horsepower', 'horsepower2']], (test_size, 2))
x3_test = np.reshape(data_test[['horsepower', 'horsepower2', 'horsepower3']], (test_size, 3))

#X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=2)
                                                   
# Initiate logistic regression object
lin_clf = LinearRegression(fit_intercept=True)
lin2_clf = LinearRegression(fit_intercept=True)
lin3_clf = LinearRegression(fit_intercept=True)

# Fit model. Let X_train = matrix of predictors, Y_train = matrix of variables.
reslin_clf = lin_clf.fit(x_train, y_train)
reslin2_clf = lin2_clf.fit(x2_train, y_train)
reslin3_clf = lin3_clf.fit(x3_train, y_train)

#Predicted values for training set
pred_lin = reslin_clf.predict(x_test)
pred_lin2 = reslin2_clf.predict(x2_test)
pred_lin3 = reslin3_clf.predict(x3_test)
 
print(np.mean(np.power(y_test - pred_lin, 2)))
print(np.mean(np.power(y_test - pred_lin2, 2)))
print(np.mean(np.power(y_test - pred_lin3, 2)))

#Calculated mean error on validation sets#
def mean_cv_err(x_data, y_data, cvobj, regobj):

    cv_errs = []

    for train_idx, test_idx in cvobj.split(x_data):

        xtrain, xtest = x_data[train_idx], x_data[test_idx]
        ytrain, ytest = y_data[train_idx], y_data[test_idx]

        res_reg = regobj.fit(xtrain, ytrain)

        pred_reg = res_reg.predict(xtest)

        cv_errs.append(np.mean(np.power(ytest - pred_reg, 2)))
    
    mean_err_out = np.mean(cv_errs)
    print('Mean error:')
    print(mean_err_out)

    return mean_err_out


#LOOCV strategy#
def loocv_err(x_data, y_data):
    
    #Leave One Out Cross-validation#
    loo = LeaveOneOut()

    llreg = LinearRegression()

    return mean_cv_err(x_data, y_data, loo, llreg)

#10-fold CV strategy#
def kfold_err(x_data, y_data):
    
    #Kfold Cross-validation#
    kfcv = KFold(n_splits=10)

    klreg = LinearRegression()

    return mean_cv_err(x_data, y_data, kfcv, klreg)


#Splitting the data for train/test#
data_size = len(data.index)

#Polynomial order#
poly_ord = 5

#Columns to use: polynomials#
poly_cols = ['horsepower']

order = 1

while True:

    x_data = np.array(np.reshape(data[poly_cols], (data_size, order)))
    y_data = np.array(np.reshape(data['mpg'], (data_size, 1)))
    
    print('\n\nPolynomial order: ' + str(order))
    print('LOOCV')
    looerr = loocv_err(x_data, y_data)
    
    print('\nKFold CV')
    kfolderr = kfold_err(x_data, y_data)

    order += 1
    
    if poly_ord < 2 or order > poly_ord:

        break

    poly_hp = 'horsepower' + str(order)
    
    data[poly_hp] = np.power(data['horsepower'], order)
    
    poly_cols.append(poly_hp)

