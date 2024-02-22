from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect




# svm # Dump-truck jew-jesus
# knn # giga fredde
# RandomForest # wallenstam
# DecisionTree # Ebitch
# Naive Bayes # Mucus

hyper_params= {
    "n_estimators": np.arange(90, 110),
    "max_depth": np.arange(23, 27),
    "min_samples_split": np.arange(1, 5),
}

df = pd.read_csv('data.csv')

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

print(df['Cover_Type'].value_counts())

target = df.pop('Cover_Type')
features = df
#features = GenericUnivariateSelect(mode='fdr').fit_transform(features, target)

#features = pd.DataFrame(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)



RF_clf = RandomForestClassifier(max_depth=25, random_state=42)
RF_clf.fit(X_train, y_train)

#grid_search = RandomizedSearchCV(RF_clf,hyper_params,n_iter=100, n_jobs=-1)
#grid_search.fit(X_train,y_train)

feature_scores = pd.Series(RF_clf.feature_importances_, index = X_train.columns).sort_values(ascending= False)

print(f"number cof columns before drop:", {len(df.columns)})
# Drops 25 columns, when score < 0.005
for attribute, score in feature_scores.items():
    if(score < 0.005):
        df = df.drop(attribute,axis = 1)
        # print(f"Dropped column: {attribute}")

print(f"number cof columns before drop after drop:", {len(df.columns)})

#y_pred_grid = grid_search.predict(X_test)
y_pred = RF_clf.predict(X_test)

print("accuracy not grid:",RF_clf.score(X_test, y_test))
#print("accuracy with grid",grid_search.score(X_test, y_test))
#print(grid_search.best_params_)

