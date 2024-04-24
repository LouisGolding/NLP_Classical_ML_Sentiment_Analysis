from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from bs4 import BeautifulSoup
from textblob import TextBlob
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
import re
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
import optuna
from sklearn.model_selection import cross_val_score



rotten_dataset = load_dataset("rotten_tomatoes")  # Loading the dataset


print(rotten_dataset["train"][:3]) # Showing first examples in the training split

# Converting to pandas df
rotten_train = rotten_dataset['train'].to_pandas()
rotten_validation = rotten_dataset['validation'].to_pandas()

rotten_test = rotten_dataset['test'].to_pandas()

# concatenating all our data except test

df_train = pd.concat([rotten_train, rotten_validation], ignore_index = True)


## EDA

df_train['label'] = df_train['label'].replace({'positive': 1, 'negative': 0})



df_train.head(5)
print(df_train['label'].value_counts())

df_train.info()
df_train.isna().sum()

## This is for when we added data, and also to check there are no duplicates in our rotten tomatoes data. 
# Printing similar rows, in case some of the added data was already present in rotten tomatoes. 

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_train['text'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

threshold = 0.9

# Find pairs with similarity above the threshold
similar_pairs = []
for i in range(len(cosine_sim)):
    for j in range(i+1, len(cosine_sim)):
        if cosine_sim[i, j] > threshold:
            similar_pairs.append((i, j))

print("\nSimilar Pairs (Cosine Similarity > {}):".format(threshold))
for pair in similar_pairs:
    print(df_train.loc[pair[0]], "\n")
    print(df_train.loc[pair[1]], "\n")


rows_to_omit = set()
for i in range(len(cosine_sim)):
    for j in range(i+1, len(cosine_sim)):
        if cosine_sim[i, j] > threshold:
            rows_to_omit.add(i)
            rows_to_omit.add(j)

# Omit similar rows
df_train = df_train[~df_train.index.isin(rows_to_omit)].reset_index(drop=True)

# Display the filtered DataFrame
print(df_train)


## Preprocessing

class LinguisticPreprocessor(TransformerMixin):
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self._remove_html_tags(X)
        X = self._remove_all_punctuations(X)
        X = self._remove_double_spaces(X)
        X = self._lemmatize(X)
        X = self._remove_capital_letters(X)
        return X

    def _remove_html_tags(self, X):
        X = list(map(lambda x: BeautifulSoup(x, 'html.parser').get_text(), X))
        return X

    def _remove_all_punctuations(self, X):
        X = list(
            map(
                lambda text: re.sub('[%s]' % re.escape(string.punctuation), '', text),
                X
            )
        )
        return X

    def _remove_double_spaces(self, X):
        X = list(map(lambda text: re.sub(" +", " ", text), X))
        return X

    def _lemmatize(self, X):
        X = list(map(lambda text: self._lemmatize_one_sentence(text), X))
        return X

    def _lemmatize_one_sentence(self, sentence):
        sentence = nltk.word_tokenize(sentence)
        sentence = list(map(lambda word: self.lemmatizer.lemmatize(word), sentence))
        return " ".join(sentence)

    def _remove_capital_letters(self, X):
        X = list(map(lambda text: text.lower(), X))
        return X



## Train/test defining



X_train = df_train['text']
y_train = df_train['label']

X_test = rotten_test['text']
y_test = rotten_test['label']



processor = LinguisticPreprocessor()



## Model

# logic of this model: we're interested in finding the probability of a class given some features.
# Litteraly conditional probabilities. 
# Text classification works well with NB apparently. 


def objective(trial):
    alpha = trial.suggest_float('alpha', 0.01, 2.0)
    fit_prior = trial.suggest_categorical('fit_prior', [True, False])
    class_prior = trial.suggest_categorical('class_prior', [None, [0.2, 2.0]])
    
    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
    
    pipeline = Pipeline(
        steps=[
            ("processor", LinguisticPreprocessor()),
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 1))),
            ("model", model)
        ]
    )
    
    score = cross_val_score(pipeline, X_train, y_train, scoring='f1_macro', cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
print("Best parameters:", best_params)

best_model = MultinomialNB(alpha=best_params['alpha'], 
                           fit_prior=best_params['fit_prior'], 
                           class_prior=best_params['class_prior'])

pipeline = Pipeline(
    steps=[
        ("processor", LinguisticPreprocessor()),
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 1))),
        ("model", best_model)
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
f1_macro = f1_score(y_test, y_pred, average="macro")
print("F1 macro score on test set: {:.2f}".format(f1_macro))








