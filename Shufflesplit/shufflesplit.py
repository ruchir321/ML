#%%
import numpy as np
from sklearn.model_selection import ShuffleSplit

#%%
X = np.array([[1,2],[3,4],[5,6],[7,8]])
Y  = np.array([1,2,1,2])

# %%
splitdata = ShuffleSplit(n_splits=5, test_size=.25,random_state=0)
splitdata.get_n_splits(X)
print(X)
# %%
print(splitdata)
# %%
for train_idx, test_idx in splitdata.split(X):
    print("TRAINSET",train_idx,"TESTSET",test_idx)
# %%
