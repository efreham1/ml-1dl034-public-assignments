from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data.csv', index_col=0)

target = df.pop('Cover_Type')
features = df

X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

clf = RandomForestClassifier(max_depth=8, random_state=42)

clf.fit(X_train, y_train)

feature_scores = pd.Series(clf.feature_importances_, index = X_train.columns).sort_values(ascending= False)

# Drops 25 columns, when score < 0.005
for attribute, score in feature_scores.items():
    if(score < 0.005):
        df = df.drop(attribute,axis = 1)
        print("Edvin är gay")
        # print(f"Dropped column: {attribute}")
 





sns.barplot(x=feature_scores, y=feature_scores.index, )
plt.title("Importance of each Feature for the Random Forest classifier")
plt.xlabel('Importance')
plt.ylabel('Features')
plt.savefig("penis")