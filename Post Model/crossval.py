#%%
from  __future__ import print_function
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

# %%
boston = load_boston()
X = boston.data
Y = boston.target
# %%
for name, met in [
    ('linear_regression', LinearRegression())
]:
    #Fit on the whole data:
    met.fit(X, Y)

    #Predict on the whole data
    p = met.predict(X)
    r2_train = r2_score(Y, p)

    #Now we use 10 fold cross-validation to estimate generalization error
    kf = KFold(len(X))
    p = np.zeros_like(Y)
    for train, test in kf:
        met.fit(x[train], Y[train])
        p[test] = met.predict(x[test])
    
    r2_cv = r2_score(Y, p)
    print('Method: {}'.format(name))
    print('Training: {}'.format(r2_train))
    print('10-fold CV: {}'.format(r2_cv))
# %%
