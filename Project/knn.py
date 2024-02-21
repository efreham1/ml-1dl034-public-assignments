from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

hyper_params = {
    "algorithm": ['ball_tree', 'kd_tree'],
    "leaf_size": np.arange(1, 10),
    "p": np.arange(1, 10),
    "n_neighbors": np.arange(5, 20)
}

df = pd.read_csv('data.csv', index_col=0)

target = df.pop('Cover_Type')
features = df

X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
clf = KNeighborsClassifier(9, weights='distance')

#clf = RandomizedSearchCV(clf, hyper_params, n_iter=150, n_jobs=-1)
clf = GridSearchCV(clf, hyper_params, n_jobs=-1)

clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.score(X_test, y_test))