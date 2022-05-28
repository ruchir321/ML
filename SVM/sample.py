#%%
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
#%%
from sklearn.preprocessing import StandardScaler
#%%
from sklearn.model_selection import train_test_split
#%%
from sklearn.svm import SVC
#%%
sampledataset = datasets.load_iris()
#%%
X = sampledataset.data[:,[2,3]]
Y = sampledataset.target
print('labels:',np.unique(Y))
#%%
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
# %%
clasification = SVC(kernel='linear',C=1.0,random_state=0)
clasification.fit(X_train,Y_train)
# %%
accuracy = metrics.accuracy_score(Y_train,clasification.predict(X_train))