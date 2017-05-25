from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from pandas.tools.plotting import scatter_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm

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

#Create the binary variable CRIM01#
data['CRIM01'] = np.where(data['CRIM'] > data['CRIM'].median(), 1, 0)

#Columns#
numcols = list(data.columns)
numcols.remove('CRIM')

#Predictor without target columns#
xcols = list(numcols)
xcols.remove('CRIM01')

#Summary (mean, stdev, range, etc)#
print('\nFull data summary')
print(data.describe())

#Correlations#
print('\nData correlations')
dcorrs = data.corr()
print(dcorrs)

#Pair plot matrix#
#sns.set()
#sns.pairplot(data[numcols], hue='CRIM01')


print('\n\n### LOGISTIC REGRESSION###')

## Logistic regression with statsmodels ##
plr_form = 'CRIM01~' + '+'.join(xcols)
prelogreg = smf.glm(formula=plr_form, data=data, family=sm.families.Binomial()).fit()

#Remove predictors with high P-values from LogReg#
logit_pvals = prelogreg.pvalues
pred_keep = list(logit_pvals[logit_pvals < 0.05].index)
pred_keep.remove('Intercept')
print('\nAfter first LogReg iteration, keeping only: ', pred_keep)

# New LogReg with only low p-value predictors#
lr_form = 'CRIM01~' + '+'.join(pred_keep)
logreg = smf.glm(formula=lr_form, data=data, family=sm.families.Binomial()).fit()

print('\nLogistic regression fit summary')
print(logreg.summary())

#Splitting the data for train/test#
X_data = data[pred_keep]
Y_data = data['CRIM01']

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.5, random_state=42, stratify=Y_data)

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
print("Precision LDA:", np.round(precision_score(Y_test, Y_pred_lda, pos_label=1), 3))
print("Recall LDA:", np.round(recall_score(Y_test, Y_pred_lda, pos_label=1), 3))



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
print("Precision QDA:", np.round(precision_score(Y_test, Y_pred_qda, pos_label=1), 3))
print("Recall QDA:", np.round(recall_score(Y_test, Y_pred_qda, pos_label=1), 3))



print('\n\n### K NEAREST NEIGHBORS ###')
#K value#
kval = 15
print('\nUsing k = ' + str(kval))

# Initiate KNN object
knn_clf = KNeighborsClassifier(n_neighbors=15)

# Fit model. Let X_train = matrix of new_pred, Y_train = matrix of variables.
resknn_clf = knn_clf.fit(X_train, Y_train)

#Predicted values for training set
Y_pred_knn = resknn_clf.predict(X_test)

#Confusion matrix#
print("\nConfusion matrix KNN:")
print(confusion_matrix(Y_test, Y_pred_knn))

#Accuracy, precision and recall#
print("\nAccuracy KNN:", np.round(accuracy_score(Y_test, Y_pred_knn), 3))
print("Precision KNN:", np.round(precision_score(Y_test, Y_pred_knn, pos_label=1), 3))
print("Recall KNN:", np.round(recall_score(Y_test, Y_pred_knn, pos_label=1), 3))

#plt.show()
