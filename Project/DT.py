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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

hParams1 = {    
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

hParams2 = {    
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],    
    "splitter": ["best", "random"],
    "min_weight_fraction_leaf": [0.0, 0.1, 0.2],
    "max_leaf_nodes": [None, 5, 10, 20],
    "min_impurity_decrease": [0.0, 0.1, 0.2]    
}

def load_dataset():
    df_ = pd.read_csv('train_dataset.csv')
    forest_ = tree.DecisionTreeClassifier(random_state=42)
    target_ = df_.pop(' Forest Cover Type Classes')

    return df_, forest_, target_

def calc_score(df,forest,target):
    global X_train, X_test, y_train, y_test
    features = df

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)

    forest_score = cross_val_score(forest, X_test, y_test, cv=10)
    # print("Decision tree %0.2f accuracy with a standard deviation of %0.2f" % (forest_score.mean(), forest_score.std()))

    forest.fit(X_train, y_train)
    return forest_score.mean(), forest_score.std()

    
def calc_score2(df,forest,target):
    global X_train, X_test, y_train, y_test
    features = df
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
    # får ej det här att funka (nedan)
    forest = GridSearchCV(forest, hParams1, n_jobs=-1)

    forest.fit(X_train, y_train)    
    print(forest.score(X_test, y_test))
    
    return forest.score()

def calc_score3(df,forest,target):
    global X_train, X_test, y_train, y_test
    features = df
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
    forest.fit(X_train,y_train)
    return forest.score(X_test,y_test)

def drop_columns(df,score_threshold,forest):
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

def random_Search(df, forest,target,params):
    
    global X_train, X_test, y_train, y_test
    features = df
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
    
    # random_search = RandomizedSearchCV(forest, param_distributions=params, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    random_search = RandomizedSearchCV(forest,param_distributions=params)

    random_search.fit(X_train, y_train) 
    
    return random_search.best_params_ , random_search.best_score_    

def grid_Search(df, forest,target,params):
    global X_train, X_test, y_train, y_test
    features = df
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
    
    gs = GridSearchCV(forest,param_grid=params,n_jobs=-1)

    gs.fit(X_train, y_train)  
    
    return gs.best_params_ , gs.best_score_        
    
def main():
    
    print("loading dataset...")
    df, forest, target = load_dataset()    
    print("dataset loaded!")
    
    print()
    
    print("running random search...")
    best_params , best_score = random_Search(df,forest,target,hParams2)
    print("random search complete")
    print("Best parameters found: ", best_params)
    print("Best accuracy found: ", best_score)   
    
    print()
                
    print("reloading dataset...")
    df, forest, target = load_dataset()    
    print("dataset loaded!")
    
    print("")
    
    print("running grid search...")
    best_params , best_score = grid_Search(df,forest,target,hParams1)
    print("grid search complete")
    print("Best parameters found: ", best_params)
    print("Best accuracy found: ", best_score)   
    
    print("reloading dataset...")
    df, forest, target = load_dataset()    
    print("dataset loaded!")
    
    print("")
    
    print("calculating score...")
    print(f"forest.score : {calc_score3(df,forest,target)}")
    
    print("")
    
    print("reloading dataset...")
    df, forest, target = load_dataset()    
    print("dataset loaded!")
    
    print("")
    
    print("calculating mean and dev...")
    mean, dev = calc_score(df,forest,target)    
    print(f"mean: {mean}, dev: {dev}")    
    
    exit()
    
    start = 0.000001 
    end = 0.00001
    step = 0.000001
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
        sys.stdout.flush()
        sys.stdout.write("\r")
        sys.stdout.flush()
        # ---- Loading Indicator ----
    
        # ---- Actual work ----
        df, forest, target = load_dataset()
        calc_score(df,forest,target)
        drop_columns(df,score_threshold,forest)
        mean, dev = calc_score(df,forest,target)
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
    print(f"best_threshold_dev = {'{:.10f}'.format(best_threshold_dev)}")
    print(f"best_threshold_mean = {'{:.10f}'.format(best_threshold_mean)}") 

if __name__ == "__main__" : main()