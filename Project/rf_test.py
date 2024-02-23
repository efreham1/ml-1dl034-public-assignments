import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data.csv')
target = df.pop('Cover_Type')
features = df

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train initial model
RF_clf = RandomForestClassifier(max_depth=25, random_state=42)
RF_clf.fit(X_train, y_train)

# Calculate feature importances
feature_scores = pd.Series(RF_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Identify features to drop (threshold can be adjusted)
threshold = 0.001
features_to_drop = feature_scores[feature_scores < threshold].index

# Drop features from train and test sets
X_train_new = X_train.drop(features_to_drop, axis=1)
X_test_new = X_test.drop(features_to_drop, axis=1)

# Re-train model with selected features
RF_clf.fit(X_train_new, y_train)

# Evaluation
accuracy = RF_clf.score(X_test_new, y_test)
print("Accuracy with selected features:", accuracy)

# Optional: Visualizing feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.show()
