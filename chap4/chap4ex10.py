from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import statsmodels.formula.api as smf
import statsmodels.api as sm

filename = '../Weekly.csv'

data = pd.read_csv(filename)

#All predictors#
allpred = list(data.columns)
allpred.remove('Direction')

#Summary (mean, stdev, range, etc)#
print('\nFull data summary')
print(data.describe())

#Correlations#
print('\nData correlations')
print(data.corr())

#List of predictors#
predictors = list(allpred)
predictors.remove('Year')
predictors.remove('Today')

#Pair plot matrix#
sns.set()
sns.pairplot(data, hue='Direction')


print('\n\n### LOGISTIC REGRESSION###')

## Logistic regression with statsmodels ##
lr_form = 'Direction~' + '+'.join(predictors)
logreg = smf.glm(formula=lr_form, data=data, family=sm.families.Binomial()).fit()

print('\nLogistic regression fit summary')
print(logreg.summary())

## Logistic regression with sklearn ##
#Prepare data#
X_full = np.array(data[predictors])
Y_full = np.array(data['Direction'])

# Initiate logistic regression object
logit = LogisticRegression()

# Fit model. Let X_full = matrix of predictors, y_train = matrix of variables.
# NOTE: Do not include a column for the intercept when fitting the model.
resLogit = logit.fit(X_full, Y_full)

#Predicted values for training set
Y_pred_full = resLogit.predict(X_full)

#Confusion matrix#
print("\nConfusion matrix full:")
print(confusion_matrix(Y_full, Y_pred_full))

#Accuracy, precision and recall#
print('\nAccuracy full:', np.round(accuracy_score(Y_full, Y_pred_full), 3))
print("Precision full:", np.round(precision_score(Y_full, Y_pred_full, pos_label='Up'), 3))
print("Recall full:", np.round(recall_score(Y_full, Y_pred_full, pos_label='Up'), 3))

## Keeping a test set based in year ##
print('\n\nUsing train/test set')

new_pred = ['Lag2']

#Prepare data#
data_train = data[data['Year'] < 2009]
data_test = data[data['Year'] >= 2009]

X_train = np.array(data_train[new_pred])
Y_train = np.array(data_train['Direction'])

X_test = np.array(data_test[new_pred])
Y_test = np.array(data_test['Direction'])

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
print("Precision logit:", np.round(precision_score(Y_test, Y_pred, pos_label='Up'), 3))
print("Recall logit:", np.round(recall_score(Y_test, Y_pred, pos_label='Up'), 3))


print('\n\n### LINEAR DISCRIMINANT ANALYSIS ###')
# Initiate logistic regression object
lda_clf = LinearDiscriminantAnalysis()

# Fit model. Let X_train = matrix of new_pred, Y_train = matrix of variables.
reslda_clf = lda_clf.fit(X_train, Y_train)

#Predicted values for training set
Y_pred_lda = reslda_clf.predict(X_test)

#Prior probabilities#
print("\nPrior probabilities")
print(reslda_clf.classes_)
print(reslda_clf.priors_)

#Group means#
print("\nGroup means")
#print(reslda_clf.classes_)
print(reslda_clf.means_)

#Coefficients#
print("\nCoefficients")
#print(reslda_clf.classes_)
print(reslda_clf.intercept_)
print(reslda_clf.coef_)

#Confusion matrix#
print("\nConfusion matrix LDA:")
print(confusion_matrix(Y_test, Y_pred_lda))

#Accuracy, precision and recall#
print("\nAccuracy LDA:", np.round(accuracy_score(Y_test, Y_pred_lda), 3))
print("Precision LDA:", np.round(precision_score(Y_test, Y_pred_lda, pos_label='Up'), 3))
print("Recall LDA:", np.round(recall_score(Y_test, Y_pred_lda, pos_label='Up'), 3))


print('\n\n### QUADRATIC DISCRIMINANT ANALYSIS ###')
# Initiate QDA object
qda_clf = QuadraticDiscriminantAnalysis()

# Fit model. Let X_train = matrix of new_pred, Y_train = matrix of variables.
resqda_clf = qda_clf.fit(X_train, Y_train)

#Predicted values for training set
Y_pred_qda = resqda_clf.predict(X_test)

#Prior probabilities#
print("\nPrior probabilities")
print(resqda_clf.classes_)
print(resqda_clf.priors_)

#Group means#
print("\nGroup means")
#print(resqda_clf.classes_)
print(resqda_clf.means_)

#Confusion matrix#
print("\nConfusion matrix QDA:")
print(confusion_matrix(Y_test, Y_pred_qda))

#Accuracy, precision and recall#
print("\nAccuracy QDA:", np.round(accuracy_score(Y_test, Y_pred_qda), 3))
print("Precision QDA:", np.round(precision_score(Y_test, Y_pred_qda, pos_label='Up'), 3))
print("Recall QDA:", np.round(recall_score(Y_test, Y_pred_qda, pos_label='Up'), 3))



print('\n\n### K NEAREST NEIGHBORS ###')
# Initiate KNN object
knn_clf = KNeighborsClassifier(n_neighbors=1)

# Fit model. Let X_train = matrix of new_pred, Y_train = matrix of variables.
resknn_clf = knn_clf.fit(X_train, Y_train)

#Predicted values for training set
Y_pred_knn = resknn_clf.predict(X_test)

#Confusion matrix#
print("\nConfusion matrix KNN:")
print(confusion_matrix(Y_test, Y_pred_knn))

#Accuracy, precision and recall#
print("\nAccuracy KNN:", np.round(accuracy_score(Y_test, Y_pred_knn), 3))
print("Precision KNN:", np.round(precision_score(Y_test, Y_pred_knn, pos_label='Up'), 3))
print("Recall KNN:", np.round(recall_score(Y_test, Y_pred_knn, pos_label='Up'), 3))


#plt.show() 



