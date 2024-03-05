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
    "weights": ['uniform', 'distance'],
    "leaf_size": np.arange(1, 15),
    "p": np.arange(1, 10),
    "n_neighbors": np.arange(4, 20)
}

df = pd.read_csv('train_dataset.csv')

target = df.pop(' Forest Cover Type Classes')
features = df

scaler = preprocessing.StandardScaler()

for col in ["Elevation (meters)", " Aspect (azimuth)", " Slope (degrees)", " Horizontal_Distance_To_Hydrology (meters)", " Vertical_Distance_To_Hydrology (meters)", " Horizontal_Distance_To_Roadways(meters)", " Hillshade_9am (0-255)", " Hillshade_Noon (0-255)", " Hillshade_3pm (0-255)", " Horizontal_Distance_To_Fire_Points (meters)"]:
    features[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(features[col])),columns=[col])

features = GenericUnivariateSelect(mode='fdr').fit_transform(features, target)

X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
clf = KNeighborsClassifier(n_jobs=-1)

#clf = RandomizedSearchCV(clf, hyper_params, n_iter=10, n_jobs=-1)
#clf = GridSearchCV(clf, hyper_params, n_jobs=-1)

clf.fit(X_train, y_train)
#print(clf.best_params_)
print(clf.score(X_test, y_test))