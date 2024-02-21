from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform




df = pd.read_csv('data.csv', index_col=0)

target = df.pop('Cover_Type')
features = df

X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)


hyper_params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

clf = GaussianNB()

random_search = RandomizedSearchCV(clf, hyper_params, n_iter=150, n_jobs=-1)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best hyperparameters:", best_params)
print("Best cross-validation score:", best_score)


    
