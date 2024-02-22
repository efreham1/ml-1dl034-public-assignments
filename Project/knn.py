from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.feature_selection import GenericUnivariateSelect
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

df = pd.read_csv('data.csv')

target = df.pop('Cover_Type')
features = df

scaler = preprocessing.StandardScaler()

for col in ['Elevation','Aspect','Slope','R_Hydrology','Z_Hydrology','R_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','R_Fire_Points']:
    features[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(features[col])),columns=[col])

features = GenericUnivariateSelect(mode='fdr').fit_transform(features, target)

X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
clf = KNeighborsClassifier(7, weights='distance', algorithm='ball_tree', leaf_size= 3, p= 1)

#clf = RandomizedSearchCV(clf, hyper_params, n_iter=150, n_jobs=-1)
#clf = GridSearchCV(clf, hyper_params, n_jobs=-1)

clf.fit(X_train, y_train)
#print(clf.best_params_)
print(clf.score(X_test, y_test))