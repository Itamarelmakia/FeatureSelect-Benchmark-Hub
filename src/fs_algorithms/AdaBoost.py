from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os
import random as rn
import pandas as pd


import os
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn
from packaging.version import parse as V

def AdaBoost_FS(X_train, y_train, k, row, random_state):
    """
    Select features using the AdaBoost classifier with a Decision Tree base estimator. This function returns the indices
    of the top 'k' important features based on their importance scores.

    Args:
        X_train (array or DataFrame): Training data with features.
        y_train (array or Series): Labels corresponding to the training data.
        k (int): Number of top features to select.
        row (dict): Hyperparameters:
            - max_depth (int)
            - n_estimators (int)
            - learning_rate (float)
        random_state (int): Seed for reproducibility.

    Returns:
        list: Indices of the top 'k' important features.
    """

    # reproducibility
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    rn.seed(random_state)

    # how many classes
    n_classes = np.unique(y_train).size

    # choose algorithm variant based on sklearn version
    skl_v = V(sklearn.__version__)
    # before v1.5.0 AdaBoost supported the 'SAMME.R' real‐valued algorithm in binary cases;
    # in newer versions only 'SAMME' is allowed.
    if skl_v < V("1.5.0") and n_classes == 2:
        algorithm = 'SAMME.R'
    else:
        algorithm = 'SAMME'

    # initialize and fit
    abc = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=int(row['max_depth'])),
        n_estimators=int(row['n_estimators']),
        learning_rate=float(row['learning_rate']),
        algorithm=algorithm,
        random_state=random_state
    )
    abc.fit(X_train, y_train)

    # feature importances → top k indices
    importances = abc.feature_importances_
    indices = list(np.argsort(importances)[-k:])

    return indices






    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    
    
    # Get feature importances
    importances = abc.feature_importances_
    
    # Normalize importances based on all features
    normalized_importances = importances / np.sum(importances)

    # Get indices of the top 'k' important features in descending order based on normalized importances
    indices = np.argsort(-normalized_importances)[:k]

    # Accumulated importance of the normalized importances
    accumulated_importance = np.round(np.sum(normalized_importances[indices]), 3)
    if accumulated_importance > 1:  # Correct if sum exceeds 1 due to rounding
        accumulated_importance = 1

    # Extract the normalized scores for the top 'k' features for output
    top_k_normalized_importances = normalized_importances[indices]

    return list(indices), list(np.round(top_k_normalized_importances, 3)), accumulated_importance
    """