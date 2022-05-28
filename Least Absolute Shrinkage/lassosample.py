'''
Lasso regression helps create simpler model with less variance
The higher weighed variables are penalised.
Cost(w)=RSS(w)+α*(sum of absolute weights)
It uses L1 regularization

Boston Dataset has 13 numerical features and 1 numerical target variable
'''

from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

datasetsample = load_boston()
X = datasetsample.data
Y = datasetsample.target

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

regr = Lasso(alpha=1.5)

mdl = regr.fit(X_std,Y)

y_pred = regr.predict(X_std)