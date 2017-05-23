from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from pandas.plotting import scatter_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm

filename = '../Smarket.csv'

data = pd.read_csv(filename)

#Convert to NaN#
#data = rawdata.replace(to_replace='?', value=np.nan).copy()
#Convert to NaN#
#data = rawdata.replace(to_replace='None', value=np.nan).copy()

#All columns#
print('\nPredictors')
print(data.columns)

#All predictors#
allpred = list(data.columns)
allpred.remove('Direction')

#Dimensions of dataframe#
print('\nDimensions of data')
print(data.shape)

#Summary (mean, stdev, range, etc)#
print('\nFull data summary')
print(data.describe())

#Correlations#
print('\nData correlations')
print(data.corr())

#plot Volumen vs Year#
figv, axv = plt.subplots()
axv.scatter(data['Year'], data['Volume'])
axv.set_xlabel('Year')
axv.set_ylabel('Volume')
axv.legend()
 
#List of predictors#
predictors = list(allpred)
predictors.remove('Year')
predictors.remove('Today')

#Pair plot matrix#
sns.set()
sns.pairplot(data, hue='Direction')
#fig_scatter = scatter_matrix(data)


### LOGISTIC REGRESSION###

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

# Calculate matrix of predicted class probabilities. 
# Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
predProbs = np.matrix(resLogit.predict_proba(X_full))

# Design matrix -- add column of 1's at the beginning of your X_full matrix
X_design = np.hstack((np.ones(shape = (X_full.shape[0],1)), X_full))

# Initiate matrix of 0's, fill diagonal with each predicted observation's variance
V = np.matrix(np.zeros(shape = (X_design.shape[0], X_design.shape[0])))
np.fill_diagonal(V, np.multiply(predProbs[:,0], predProbs[:,1]).A1)

# Covariance matrix
covLogit = np.linalg.inv(X_design.T * V * X_design)
# All fit parameters
logitParams = np.insert(resLogit.coef_, 0, resLogit.intercept_)
# Standard errors
std_errors = np.sqrt(np.diag(covLogit))
# Z statistic (coefficient / s.e.)#
zscores = logitParams / std_errors
print('\nParameters: (Intercept) , ' + ' , '.join(predictors))
print("LogReg parameters: ", np.round(logitParams, 2))
print("Standard errors: ", np.round(std_errors, 2))
print("z-statistics: ", np.round(zscores, 2))

#P-values#
p_values = scipy.stats.norm.sf(abs(zscores))*2
print("p-values: ", np.round(p_values, 2))

#Predict probabilities first 10 values#
print("\n Predited probabilities for train data: ", np.round(predProbs[1:10], 3))

#Predicted values for training set
Y_pred_full = resLogit.predict(X_full)

#Confusion matrix#
print("\n Confusion matrix:")
print(confusion_matrix(Y_full, Y_pred_full))

#Accuracy, precision and recall#
print('\n Accuracy:', np.round(accuracy_score(Y_full, Y_pred_full), 3))
print("Precision:", np.round(precision_score(Y_full, Y_pred_full, pos_label='Up'), 3))
print("Recall:", np.round(recall_score(Y_full, Y_pred_full, pos_label='Up'), 3))

## Keeping a test set based in year ##
print('\n\nUsing train/test set')

#Prepare data#
data_train = data[data['Year'] < 2005]
data_test = data[data['Year'] >= 2005]

X_train = np.array(data_train[predictors])
Y_train = np.array(data_train['Direction'])

X_test = np.array(data_test[predictors])
Y_test = np.array(data_test['Direction'])

# Initiate logistic regression object
logit_clf = LogisticRegression()

# Fit model. Let X_train = matrix of predictors, Y_train = matrix of variables.
resLogit_clf = logit_clf.fit(X_train, Y_train)

#Predicted values for training set
Y_pred = resLogit_clf.predict(X_test)

#Confusion matrix#
print("\n Confusion matrix:")
print(confusion_matrix(Y_test, Y_pred))

#Accuracy, precision and recall#
print('\n Accuracy:', np.round(accuracy_score(Y_test, Y_pred), 3))
print("Precision:", np.round(precision_score(Y_test, Y_pred, pos_label='Up'), 3))
print("Recall:", np.round(recall_score(Y_test, Y_pred, pos_label='Up'), 3))


## Now just using lag1 and lag2 ##
print('\n\nUsing only Lag1 and Lag2 as predictors/features')

new_pred = ['Lag1', 'Lag2']

Xs_train = np.array(data_train[new_pred])
Ys_train = np.array(data_train['Direction'])

Xs_test = np.array(data_test[new_pred])
Ys_test = np.array(data_test['Direction'])

print(data_train[new_pred].shape)
print(Xs_train.shape)

# Initiate logistic regression object
logit_clf2 = LogisticRegression()

# Fit model. Let Xs_train = matrix of new_pred, Ys_train = matrix of variables.
resLogit_clf2 = logit_clf2.fit(Xs_train, Ys_train)

#Predicted values for training set
Ys_pred = resLogit_clf2.predict(Xs_test)

#Confusion matrix#
print("\n Confusion matrix:")
print(confusion_matrix(Ys_test, Ys_pred))

#Accuracy, precision and recall#
print('\n Accuracy:', np.round(accuracy_score(Ys_test, Ys_pred), 3))
print("Precision:", np.round(precision_score(Ys_test, Ys_pred, pos_label='Up'), 3))
print("Recall:", np.round(recall_score(Ys_test, Ys_pred, pos_label='Up'), 3))

## Predict new values p1=(1.2, 1.1) and p2=(1.5, -0.8) ##
#Sklearn's predict_proba throws probabilities for each class. For the order check self.classes_
#In our case we are interested in the probability of 'Up', so it should be the second column
p1 = [1.2, 1.1]
p2 = [1.5, -0.8]
pn = np.vstack((p1, p2))
print(resLogit_clf2.classes_)
print(resLogit_clf2.predict_proba(pn))


### LINEAR DISCRIMINANT ANALYSIS ###
# Initiate logistic regression object
lda_clf = LinearDiscriminantAnalysis()

# Fit model. Let Xs_train = matrix of new_pred, Ys_train = matrix of variables.
reslda_clf = lda_clf.fit(Xs_train, Ys_train)

#Predicted values for training set
Ys_pred_lda = reslda_clf.predict(Xs_test)

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
print(confusion_matrix(Ys_test, Ys_pred_lda))

#Accuracy, precision and recall#
print("\nAccuracy LDA:", np.round(accuracy_score(Ys_test, Ys_pred_lda), 3))
print("Precision LDA:", np.round(precision_score(Ys_test, Ys_pred_lda, pos_label='Up'), 3))
print("Recall LDA:", np.round(recall_score(Ys_test, Ys_pred_lda, pos_label='Up'), 3))

#Prediction probabilities#
lda_probs = reslda_clf.predict_proba(Xs_test)

#Probabilities of market going up#
up_probs = lda_probs[:, 1]

#Indices of events with posterior probabilities >= 50%  and < 50%#
idx_g50 = up_probs >= 0.5
idx_l50 = up_probs < 0.5

print('\nNumber of days with probability >= 50% for market to be up:', up_probs[idx_g50].size)
print('\nNumber of days with probability < 50% for market to be up:', up_probs[idx_l50].size)

#Indices of events with posterior probability > 90%#
idx_g90 = up_probs > 0.9

print('\nNumber of days with probability > 90% for market to be up:', up_probs[idx_g90].size)


### QUADRATIC DISCRIMINANT ANALYSIS ###
# Initiate QDA object
qda_clf = QuadraticDiscriminantAnalysis()

# Fit model. Let Xs_train = matrix of new_pred, Ys_train = matrix of variables.
resqda_clf = qda_clf.fit(Xs_train, Ys_train)

#Predicted values for training set
Ys_pred_qda = resqda_clf.predict(Xs_test)

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
print(confusion_matrix(Ys_test, Ys_pred_qda))

#Accuracy, precision and recall#
print("\nAccuracy QDA:", np.round(accuracy_score(Ys_test, Ys_pred_qda), 3))
print("Precision QDA:", np.round(precision_score(Ys_test, Ys_pred_qda, pos_label='Up'), 3))
print("Recall QDA:", np.round(recall_score(Ys_test, Ys_pred_qda, pos_label='Up'), 3))

### K NEAREST NEIGHBORS ###
# Initiate KNN object
knn_clf = KNeighborsClassifier(n_neighbors=3)

# Fit model. Let Xs_train = matrix of new_pred, Ys_train = matrix of variables.
resknn_clf = knn_clf.fit(Xs_train, Ys_train)

#Predicted values for training set
Ys_pred_knn = resknn_clf.predict(Xs_test)

#Confusion matrix#
print("\nConfusion matrix KNN:")
print(confusion_matrix(Ys_test, Ys_pred_knn))

#Accuracy, precision and recall#
print("\nAccuracy KNN:", np.round(accuracy_score(Ys_test, Ys_pred_knn), 3))
print("Precision KNN:", np.round(precision_score(Ys_test, Ys_pred_knn, pos_label='Up'), 3))
print("Recall KNN:", np.round(recall_score(Ys_test, Ys_pred_knn, pos_label='Up'), 3))

plt.show() 



