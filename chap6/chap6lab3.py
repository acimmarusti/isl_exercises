from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error

#CSV file#
filename = '../Hitters.csv'

#Load raw data to pandas dataframe#
data = pd.read_csv(filename).dropna()

#Binarize columns#
data['LeagueN'] = (data['League'] == 'N').astype(int)
data['DivisionW'] = (data['Division'] == 'W').astype(int)
data['NewLeagueN'] = (data['NewLeague'] == 'N').astype(int)

#Only numerical cols#
colmask = data.dtypes != object
numcols = colmask.index[colmask == True]
xcols = list(numcols)
xcols.remove('Salary')

#Full data split#
X = np.array(data[xcols])
Y = np.array(data['Salary'])

#Keep test set#
X_train, X_test, Y_train, Y_test = train_test_split(data[xcols], data['Salary'], test_size=0.5, random_state=42)

###Principal components regression###
pca = PCA()
X_train_red = pca.fit_transform(scale(X_train))
n = len(X_train_red)

#10-fold Cross-validation object#
kfcv = KFold(n_splits=10, shuffle=True, random_state=2)

#Linear regression#
lreg = LinearRegression()

#MSE initialization to only intercept#
scores_intercept = cross_val_score(lreg, np.zeros((n,1)), y=Y_train, scoring='neg_mean_squared_error', cv=kfcv)
mse = [-np.mean(scores_intercept)]
mse_std = [np.std(scores_intercept)]

for ii in range(1, len(xcols) + 1):
    scores = cross_val_score(lreg, X_train_red[:,:ii], y=Y_train, scoring='neg_mean_squared_error', cv=kfcv)
    mse.append(-np.mean(scores))
    mse_std.append(np.std(scores))

#Test set prediction#
npc = 6
X_test_red = pca.transform(scale(X_test))[:,:npc + 1]
nlreg = LinearRegression()
nlreg.fit(X_train_red[:,:npc + 1], Y_train)
Y_test_pred = nlreg.predict(X_test_red)

print('\nPCR Test set MSE:')
print(mean_squared_error(Y_test, Y_test_pred))

##Partial least squares##
nt = len(X_train)

mse_pls = []
mse_pls_std = []

for ii in range(1, len(xcols) + 1):
    pls = PLSRegression(n_components=ii)
    scores_pls = cross_val_score(pls, scale(X_train), y=Y_train, scoring='neg_mean_squared_error', cv=kfcv)
    mse_pls.append(-np.mean(scores_pls))
    mse_pls_std.append(np.std(scores))

pls = PLSRegression(n_components=2)
pls.fit(scale(X_train), Y_train)

print('\nPLS Test set MSE:')
print(mean_squared_error(Y_test, pls.predict(scale(X_test))))

#PCR plots#
plt.figure()
plt.plot(mse, '-v')
plt.title('PCR')
plt.xlabel('Number of principal components in regression')
plt.ylabel('Train MSE mean')
plt.tight_layout()

plt.figure()
plt.plot(mse_std, '-v')
plt.title('PCR')
plt.xlabel('Number of principal components in regression')
plt.ylabel('Train MSE Stdev')
plt.tight_layout()

#PLS plots#
plt.figure()
plt.plot(mse_pls, '-v')
plt.title('PLS')
plt.xlabel('Number of principal components in regression')
plt.ylabel('Train MSE mean')
plt.tight_layout()

plt.figure()
plt.plot(mse_pls_std, '-v')
plt.title('PLS')
plt.xlabel('Number of principal components in regression')
plt.ylabel('Train MSE Stdev')
plt.tight_layout()


plt.show()
