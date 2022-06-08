#%%
from sklearn.datasets import  make_classification
from sklearn.linear_model import LogisticRegression
# %%
from sklearn.model_selection import GridSearchCV
# %%
A, B = make_classification(1000, n_features=5)
#%%
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(A,B)
# %%
grid_search_params = {'C': [1,2,3,4]}
gsearch = GridSearchCV(logreg,grid_search_params)
gsearch.fit(A,B)
# %%
print(gsearch.cv_results_)