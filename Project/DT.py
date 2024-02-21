"""
Author: Edvin Bruce
Date: February 21, 2024
Description:
This Python script, authored by Edvin, delves into the realm of artificial intelligence and machine learning, focusing on the implementation of decision trees. Decision trees are powerful tools for classification and regression tasks, and this program showcases their usage, including model training, prediction, and performance evaluation.

Additional notes:
- The scikit-learn library is utilized for implementing decision trees.
- Proper preprocessing and formatting of the dataset are essential for optimal results.
- Experimentation with hyperparameters and feature engineering techniques is encouraged for fine-tuning the decision tree model.
"""

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
import sys

def load_dataset():
    df_ = pd.read_csv('data.csv', index_col=0)
    forest_ = tree.DecisionTreeClassifier(random_state=42)
    target_ = df_.pop('Cover_Type')

    return df_, forest_, target_

def calc_score(df):
    global X_train, X_test, y_train, y_test

    features = df

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)

    forest_score = cross_val_score(forest, X_test, y_test, cv=10)
    # print("Decision tree %0.2f accuracy with a standard deviation of %0.2f" % (forest_score.mean(), forest_score.std()))

    forest.fit(X_train, y_train)
    return forest_score.mean(), forest_score.std()

def drop_columns(df,score_threshold):
    num_columns_before_drop = len(df.columns)
    columns_dropped = 0
    feature_scores = pd.Series(forest.feature_importances_, index = X_train.columns).sort_values(ascending= False)
    for attribute, score in feature_scores.items():
        if(score < score_threshold):
            df = df.drop(attribute,axis = 1)
            columns_dropped += 1
            # print(f"Dropped column: {attribute}")

    # print(f"num columns dropped: {columns_dropped}")
    # print(f"num columns before drop: {num_columns_before_drop}")
    # print(f"num columns after drop: {len(df.columns)}")




start = 0.00001
end = 0.0001
step = 0.00001
total_steps = int((end - start) / step) + 1 # Used for loading indicator

best_mean:float = 0
best_dev:float = 1
best_threshold_dev:float = 0
best_threshold_mean:float = 0

for i, score_threshold in enumerate(np.arange(start, end + step, step)):
    # ---- Loading Indicator ----
    progress = (i + 1) / total_steps
    bar_length = int(progress * 50)
    sys.stdout.write("\r[%-50s] %d%%" % ('=' * bar_length, progress * 100)) 
    # sys.stdout.write("\r[%-50s] %d%%" % ('\U0001F40E' * bar_length, progress * 100)) # horses
    sys.stdout.flush()
    sys.stdout.write("\r")
    sys.stdout.flush()
    # ---- Loading Indicator ----

    # ---- Actual work ----
    df, forest, target = load_dataset()
    calc_score(df)
    drop_columns(df,score_threshold)
    mean, dev = calc_score(df)
    # ---- Actual work ----

    # ---- Check if we have new Highscore ----
    if best_mean < mean : 
        best_mean = mean
        best_threshold_mean = score_threshold
    if best_dev > dev : 
        best_dev = dev
        best_threshold_dev = score_threshold
    # ---- Check if we have new Highscore ----
        
print("\n")
print(f"best_mean = {best_mean}")
print(f"best_dev = {best_dev}")
print(f"best_threshold_dev = {best_threshold_dev}")
print(f"best_threshold_mean = {best_threshold_mean}")
    
