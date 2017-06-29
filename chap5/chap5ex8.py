from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import statsmodels.formula.api as smf

#Calculated mean error on validation sets#
def mean_cv_err(x_data, y_data, cvobj, regobj):

    data_size = len(x_data)

    data_shape = x_data.shape

    if len(data_shape) > 1:
        
        xdata = np.reshape(np.array(x_data), data_shape)

    else:

        xdata = np.reshape(np.array(x_data), (data_size, 1))
    
    ydata = np.reshape(y_data, (data_size, 1))
        
    
    cv_errs = []

    for train_idx, test_idx in cvobj.split(xdata):

        xtrain = xdata[train_idx]
        xtest = xdata[test_idx]
        ytrain = ydata[train_idx]
        ytest = ydata[test_idx]

        res_reg = regobj.fit(xtrain, ytrain)

        pred_reg = res_reg.predict(xtest)

        #Reshape necessary because predition produces a (1, n) numpy array, while ytest is (n, 1)#
        cv_errs.append(np.mean(np.power(np.reshape(ytest, pred_reg.shape) - pred_reg, 2)))
    
    mean_err_out = np.mean(cv_errs)

    return mean_err_out


#LOOCV strategy#
def loocv_err(x_data, y_data):
    
    #Leave One Out Cross-validation#
    loo = LeaveOneOut()

    llreg = LinearRegression()

    return mean_cv_err(x_data, y_data, loo, llreg)

#Simulated Data#
np.random.seed(1)
x = np.random.standard_normal(100)
y = x - 2 * np.square(x) + np.random.standard_normal(100)

data = pd.DataFrame()
data['y'] = y
data['x'] = x
data['x2'] = np.square(x)
data['x3'] = np.power(x, 3)
data['x4'] = np.power(x, 4)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')

#Compute LOOCV errors#
print('\nLOOCV error for linear model')
print(loocv_err(data['x'], data['y']))
print('\nLOOCV error for quadratic model')
print(loocv_err(data[['x','x2']], data['y']))
print('\nLOOCV error for cubic model')
print(loocv_err(data[['x','x2','x3']], data['y']))
print('\nLOOCV error for quartic model')
print(loocv_err(data[['x','x2','x3','x4']], data['y']))

#Linear regression#
linfit = smf.ols(formula='y ~ x + np.power(x, 2) + np.power(x, 3) + np.power(x, 4)', data=data).fit()
print(linfit.summary())

plt.show()


