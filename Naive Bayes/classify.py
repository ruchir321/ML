'''
Document classification using Naive Bayes
'''
#%%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import  Pipeline
import numpy as np
# %%
doc_train = fetch_20newsgroups(subset='train',shuffle=True)
#%%
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(doc_train.data)
# %%
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# %%
classifier = MultinomialNB().fit(X_train_tfidf, doc_train.target)
# %%
text_classifier = Pipeline([('vect', CountVectorizer()),('tfidf',TfidfTransformer())])
text_classifier = text_classifier.fit(doc_train.data, doc_train.target)
# %%
doc_test = fetch_20newsgroups(subset='test',shuffle=True)
predicted = text_classifier.predict(doc_test.data)
PredictED_mean = np.mean(predicted == doc_test.target)