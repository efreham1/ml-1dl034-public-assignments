from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')

target = df.pop('Cover_Type')
features = df

scaler = StandardScaler()

for col in ['Elevation','Aspect','Slope','R_Hydrology','Z_Hydrology','R_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','R_Fire_Points']:
    features[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(features[col])),columns=[col])

features = GenericUnivariateSelect(mode='fdr').fit_transform(features, target)

X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)

# clf = SVC(kernel='rbf')

# clf.fit(X_train, y_train)

# predict = clf.predict(X_test)

# acc = accuracy_score(y_test, predict)

# print(f"SVM acc: {100*acc:.2f}%")

clf = SVC(kernel='rbf', C = 100, gamma=0.1)

clf.fit(X_train,y_train)

scaled_acc = clf.score(X_test, y_test)

print(f"SVM scaled acc: {100*scaled_acc:.2f}%")
