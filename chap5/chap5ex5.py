from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm

#Calculated mean error on validation sets#
def get_cv_err(x_data, y_data, cvobj, regobj):

    cv_errs = []

    for train_idx, test_idx in cvobj.split(x_data):

        xtrain, xtest = x_data[train_idx], x_data[test_idx]
        ytrain, ytest = y_data[train_idx], y_data[test_idx]

        res_reg = regobj.fit(xtrain, ytrain)

        pred_reg = res_reg.predict(xtest)

        #Reshape necessary because predition produces a (1, n) numpy array, while ytest is (n, 1)#
        cv_errs.append(np.mean(np.power(np.reshape(ytest, pred_reg.shape) - pred_reg, 2)))
    
    return cv_errs


#K-fold CV strategy#
def kfold_err(x_data, y_data, num_splits=10):
    
    #Kfold Cross-validation#
    kfcv = KFold(n_splits=num_splits)

    klreg = LogisticRegression()

    return get_cv_err(x_data, y_data, kfcv, klreg)

filename = '../Default.xlsx'

#Load data to pandas dataframe and drop the missing values#
data = pd.read_excel(filename)

#Convert 'default' column into 1/0 binary column#
data['default10'] = np.zeros(len(data.index))
data.loc[data['default'] == 'Yes', 'default10'] = 1

#Data summary#
#print(data.describe())

print('\n\n### LOGISTIC REGRESSION###')

## Logistic regression with statsmodels ##
preds = ['income', 'balance']
lr_form = 'default10~' + '+'.join(preds)
logreg = smf.glm(formula=lr_form, data=data, family=sm.families.Binomial()).fit()

print('\nLogistic regression fit summary')
print(logreg.summary())

#Prepare data#
X_train, X_test, Y_train, Y_test = train_test_split(data[preds], data['default10'], test_size=0.5, stratify=data['default10'])

# Initiate logistic regression object
logit_clf = LogisticRegression()

# Fit model. Let X_train = matrix of predictors, Y_train = matrix of variables.
resLogit_clf = logit_clf.fit(X_train, Y_train)

#Predicted values for training set
Y_pred = resLogit_clf.predict(X_test)

#Confusion matrix#
print("\nConfusion matrix logit:")
print(confusion_matrix(Y_test, Y_pred))

#Accuracy, precision and recall#
print('\nAccuracy logit:', np.round(accuracy_score(Y_test, Y_pred), 3))
print("Precision logit:", np.round(precision_score(Y_test, Y_pred, pos_label=1), 3))
print("Recall logit:", np.round(recall_score(Y_test, Y_pred, pos_label=1), 3))

#Return error after 3 different splits#
nsplits = 3
print('\n\nValidation set errors for ' + str(nsplits) + ' splits')
print(kfold_err(np.array(data[preds]), np.array(data['default10']), num_splits=nsplits))


print('\n\n## Include student column ##')
#Convert 'student' column into 1/0 binary column#
data['student10'] = np.zeros(len(data.index))
data.loc[data['student'] == 'Yes', 'student10'] = 1

## Logistic regression with statsmodels ##
preds_new = ['income', 'balance', 'student10']
lr_form_new = 'default~' + '+'.join(preds_new)
logreg_new = smf.glm(formula=lr_form_new, data=data, family=sm.families.Binomial()).fit()

print('\nLogistic regression fit summary')
print(logreg_new.summary())

#Return error after 3 different splits#
print('\n\nValidation set errors for ' + str(nsplits) + ' splits')
print(kfold_err(np.array(data[preds_new]), np.array(data['default10']), num_splits=nsplits))
