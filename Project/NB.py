from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import GenericUnivariateSelect


df = pd.read_csv('data.csv')

target = df.pop('Cover_Type')
features = df
#features = GenericUnivariateSelect(mode='fdr').fit_transform(features, target)

X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)

hyper_params = {'var_smoothing': np.linspace(1e-11, 1, 2000)}

clf = GaussianNB(var_smoothing=0.010505252636208103)

#grid_search = GridSearchCV(clf, hyper_params, n_jobs=-1)
#grid_search.fit(X_train, y_train)

#best_params = grid_search.best_params_
#best_score = grid_search.best_score_

clf = clf.fit(X_train, y_train)
best_score = clf.score(X_test, y_test)


#print("Best hyperparameters:", best_params)
print("Best cross-validation score:", best_score)
