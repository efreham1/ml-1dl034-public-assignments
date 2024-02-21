from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv', index_col=0)

target = df.pop('Cover_Type')
features = df

X_train,X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)

clf = SVC(kernel='rbf')

clf.fit(X_train, y_train)

predict = clf.predict(X_test)

acc = accuracy_score(y_test, predict)

print(f"SVM acc: {100*acc:.2f}%")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit(X_test)


clf2 = SVC(kernel='rbf', C = 10, gamma=0.1)

clf2.fit(X_train_scaled,y_train)

predict = clf2.predict(X_test_scaled)

scaled_acc = accuracy_score(y_test,predict)


print(f"SVM acc: {100*scaled_acc:.2f}%")
# coff = pd.Series(clf.coef_[0], index=X_train.columns)
# coff_abs = coff.abs()
# 
# featured_scores = coff_abs[coff_abs >= 0.005].index
# 
# X_train_filtered = X_train[featured_scores]
# X_test_filtered = X_test[featured_scores]
# 
# clf_filtered = SVC(kernel='linear')
# clf_filtered.fit(X_train_filtered, y_train)
# 
# filtered_predict = clf_filtered.predict(X_test_filtered)
# 
# acc_filtered = accuracy_score(y_test, filtered_predict)

# print(f"SVM acc: {100*acc_filtered:.2f}%")

