import itertools
import string
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pylab as pl
import re
import multiprocessing as mp
import pickle
from collections import Counter
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

chunks=pd.read_csv('./dataset/proj_dataset_csv.csv',chunksize=10000)
dataset_list = []  # append each chunk df here

# Each chunk is in df format
for chunk in chunks:
    # perform data filtering

    # Once the data filtering is done, append the chunk to list
    dataset_list.append(chunk)

# concat the list into dataframe
electronics = pd.concat(dataset_list)

electronics['id']=[x for x in range(0,len(electronics))]

electronics=electronics.dropna(how='any')
electronics["reviewTime"] = pd.to_datetime(electronics["reviewTime"])
electronics = electronics[[ 'id','summary', 'reviewText', 'overall', 'helpful', 'reviewTime']]

def transformation(s):
    s=s.lower()
    s=s.strip()
    s=re.sub(r'\d+', '', s)
    return s
reviews = electronics['reviewText']
reviews=reviews.dropna()
reviews=reviews.apply(lambda x:transformation(x))

stops = stopwords.words('english')

def tokenize(text):
    tokenized = word_tokenize(text)
    no_punc = []
    for review in tokenized:
        line = "".join(char for char in review if char not in string.punctuation)
        no_punc.append(line)
    tokens = lemmatize(no_punc)
    return tokens


def lemmatize(tokens):
    lmtzr = WordNetLemmatizer()
    lemma = [lmtzr.lemmatize(t) for t in tokens]
    return lemma

reviews = reviews.apply(lambda x: tokenize(x))
electronics['pos_neg'] = [1 if x > 3 else 0 for x in electronics.overall]
review_text = electronics["reviewText"]

x_train, x_test, y_train, y_test = train_test_split(electronics.reviewText, electronics.pos_neg, random_state=42)

# Vectorize X_train
vectorizer = CountVectorizer(min_df=5).fit(x_train)
X_train = vectorizer.transform(x_train)


scores = cross_val_score(LogisticRegression(solver='liblinear'), X_train, y_train, cv=3)

print("Mean cross-validation accuracy: {:.3f}".format(np.mean(scores)))

logreg = LogisticRegression(C=0.1,solver='liblinear').fit(X_train, y_train)

X_test = vectorizer.transform(x_test)

log_y_pred = logreg.predict(X_test)

print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

joblib.dump(logreg, 'basic_logreg.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')
