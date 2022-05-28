#%%
import imp
from matplotlib import dates
from sklearn import metrics
from sklearn import datasets
from sklearn import tree
import numpy as np
import pandas as pd

sampledataset = datasets.load_iris()

X = sampledataset.data

# %%
X.shape
# %%
from sklearn.preprocessing import StandardScaler
# %%
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# %%
from sklearn.model_selection import train_test_split
# %%
y = sampledataset.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# %%
dtree = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train,y_train)
dtree.predict(X_train)
# %%
print("Accuracy: ",metrics.accuracy_score(y_train,dtree.predict(X_train)))

# %%
import pydot
# %%
from six import StringIO
# %%
out_data = StringIO()
tree.export_graphviz(dtree, out_file='tree.dot')
# %%
tree.export_graphviz(dtree, out_file=out_data,
                    feature_names=sampledataset.feature_names,
                    class_names=dtree.classes_.astype(int).astype(str),
                    filled=True, rounded=True,
                    special_characters=True,
                    node_ids=1)

graph = pydot.graph_from_dot_data(out_data.getvalue())
graph[0].write_pdf("dtree.pdf")
# %%
