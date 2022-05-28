'''
The Ridge regression uses L2 regularization which learns more complex patterns,
at the cost of not being robust to outliers.
'''
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

bostondataset = load_boston()
X = bostondataset.data
Y = bostondataset.target

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

ridgereg = Ridge(alpha=0.5,normalize=True)

mdl = ridgereg.fit(X_std,Y)

y_pred = ridgereg.predict(X_std)