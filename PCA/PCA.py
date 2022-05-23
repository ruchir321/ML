#This is a practice code for PCA on Iris dataset. Dimensionality reduction allows more efficient solution to an ML problem statement.
#It is recommneded to do a PCA in the preliminary stages of building a model

from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
#Scaled/normalized dataset is needed to find the covariance matrix
from sklearn.preprocessing import StandardScaler

#get the data to find the dimensionality
iris = datasets.load_iris()
X=iris.data
#The data has 4 dimensions
print(X.shape)

X_std = StandardScaler().fit_transform(X)
