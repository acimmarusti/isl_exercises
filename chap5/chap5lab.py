from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm

filename = '../Auto.csv'

#Load data to pandas dataframe and drop the missing values#
data = pd.read_csv(filename, na_values='?').dropna()

#Create the binary variable mpg01#
#data['mpg01'] = np.where(data['mpg'] > data['mpg'].median(), 1, 0)

#Random sampling#
data_train = data.sample(n=196, random_state=42)
data.loc[data.index, 'train'] = 'n'
data.loc[data_train.index, 'train'] = 'y' 
data_test = data[data['train'] == 'n']


#Numeric columns#
numcols = list(data.columns)
numcols.remove('name')
numcols.remove('mpg')
numcols.remove('train')

#Splitting the data for train/test#
X_data = data['horsepower']
Y_data = data['mpg']

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

print('\n\n### LINEAR REGRESSION###')

## Linear regression with statsmodels ##
lreg = smf.ols(formula='mpg~horsepower', data=data_train).fit()
print(np.mean(np.power(data_test['mpg'] - lreg.predict(data_test['horsepower']), 2)))

l2reg = smf.ols(formula='mpg~np.square(horsepower)', data=data_train).fit()
print(np.mean(np.power(data_test['mpg'] - l2reg.predict(data_test['horsepower']), 2)))

l3reg = smf.ols(formula='mpg~np.power(horsepower, 3)', data=data_train).fit()
print(np.mean(np.power(data_test['mpg'] - l3reg.predict(data_test['horsepower']), 2)))

#print(lreg.summary())
#print(l2reg.summary())
#print(l3reg.summary())

"""
# Initiate logistic regression object
logit_clf = LogisticRegression()

# Fit model. Let X_train = matrix of predictors, Y_train = matrix of variables.
resLogit_clf = logit_clf.fit(X_train, Y_train)

#Predicted values for training set
Y_pred_logit = resLogit_clf.predict(X_test)

#Confusion matrix#
print("\nConfusion matrix logit:")
print(confusion_matrix(Y_test, Y_pred_logit))

#Accuracy, precision and recall#
print('\nAccuracy logit:', np.round(accuracy_score(Y_test, Y_pred_logit), 3))
print("Precision logit:", np.round(precision_score(Y_test, Y_pred_logit, pos_label=1), 3))
print("Recall logit:", np.round(recall_score(Y_test, Y_pred_logit, pos_label=1), 3))
"""
