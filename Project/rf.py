from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.model_selection import RandomizedSearchCV
import multiprocessing


hyper_params= {
    "n_estimators": np.arange(95, 110, dtype=int),
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10, 15, 100],
    "min_samples_leaf":  [1, 2, 4, 6, 8],
    "max_features": ['sqrt', 'log2', None],
    "bootstrap":[True]
}

df = pd.read_csv('train_dataset.csv')

'''
#------printing class distribution-----
cmap = sns.color_palette('Set2', as_cmap=True)(np.arange(7))
plt.figure(figsize=(8,8))
plt.pie(
    df['Cover_Type'].value_counts().values,
    colors = cmap,
    labels = df['Cover_Type'].value_counts().keys(),
    autopct='%.1f%%'
    )
plt.title('Class distribution')
plt.savefig(fname="class_distribution")
#------printing class distribution-----
'''

#-----other bs

#print(df.head())

#print(df[' Forest Cover Type Classes'].value_counts())

target = df.pop(' Forest Cover Type Classes')
features = df
features = GenericUnivariateSelect(mode='fdr').fit_transform(features, target)

features = pd.DataFrame(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)


RF_clf = RandomForestClassifier(bootstrap= True, random_state=42)
RF_clf.fit(X_train, y_train)


feature_scores = pd.Series(RF_clf.feature_importances_, index = X_train.columns).sort_values(ascending= False)
print(f"number cof columns before drop:", {len(X_train.columns)})

# Identify features to drop (threshold can be adjusted)
threshold = 0.001
features_to_drop = feature_scores[feature_scores < threshold].index

# Drop features from train and test sets
X_train_new = X_train.drop(features_to_drop, axis=1)
X_test_new = X_test.drop(features_to_drop, axis=1)

print(f"number cof columns before drop after drop:", {len(X_train_new.columns)})

grid_search = RandomizedSearchCV(RF_clf,hyper_params,n_iter=15, verbose=2, cv=3)

#RF_clf.fit(X_train_new, y_train)
grid_search.fit(X_train_new,y_train)


#print("accuracy not grid:",RF_clf.score(X_test_new, y_test))
print("accuracy with grid",grid_search.score(X_test_new, y_test))
print(grid_search.best_params_)

#accuracy with grid 0.9421346244934478 {'n_estimators': 98, 'min_samples_split': 2, 'max_depth': 26}
#{'n_estimators': 106, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': 30, 'bootstrap': True}
#accuracy with grid 0.9609318807782047{'n_estimators': 95, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': None, 'max_depth': 50, 'bootstrap': True}
#accuracy with grid 0.965633387716436 {'n_estimators': 95, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 40, 'bootstrap': True}