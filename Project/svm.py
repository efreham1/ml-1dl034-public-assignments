from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import reciprocal, uniform
df = pd.read_csv('train_dataset.csv')
target = df.pop(' Forest Cover Type Classes')
features = df

scaler = StandardScaler()
for col in [
    "Elevation (meters)",
    " Aspect (azimuth)",
    " Slope (degrees)",
    " Horizontal_Distance_To_Hydrology (meters)",
    " Vertical_Distance_To_Hydrology (meters)",
    " Horizontal_Distance_To_Roadways(meters)",
    " Hillshade_9am (0-255)",
    " Hillshade_Noon (0-255)",
    " Hillshade_3pm (0-255)",
    " Horizontal_Distance_To_Fire_Points (meters)",
    " Rawah Wilderness Area (1/4)",
    " Neota Wilderness Area (2/4)",
    " Comanche Peak Wilderness Area (3/4)",
    " Cache la Poudre Wilderness Area (4/4)"
]:
    features[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(features[col])),columns=[col])
features = GenericUnivariateSelect(mode='fdr').fit_transform(features, target)


X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)


clf = LinearSVC(random_state=42)

clf.fit(X_train,y_train)
print("FYRA")

scaled_acc = clf.score(X_test, y_test)
print("FEM")

print(f"SVM scaled acc: {100*scaled_acc:.2f}%")