
"""For Univariate_FS function, as we not including any steps that might introduce randomness into the workflow (like data shuffling or stochastic modeling techniques). 
If you decide to change the logic of code flow ensure that those steps are controlled by appropriate seeding or by passing a random_state where applicable. This approach maintains the integrity and reproducibility of your analyses, keeping the deterministic nature of your feature selection process in Univariate_FS unaffected by external random factors.
"""

import random as rn
import os
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime, timedelta
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

def Univariate_FS(p_train_feature, p_train_label, key_feature_number, row,random_state):
    """
    Perform univariate feature selection based on statistical tests. This function uses SelectKBest to rank features 
    based on a user-defined score function and selects the top 'key_feature_number' features.

    Args:
        p_train_feature (numpy array): Training data with features.
        p_train_label (numpy array): Labels corresponding to the training data.
        key_feature_number (int): Number of features to select.
        row (dict): Configuration dictionary with parameters:
            - 'score_func' (str): Score function to use for feature selection. Options are 'chi2' or 'f_classif'.

    Returns:
        list: Indices of the top 'k' important features.

    Raises:
        ValueError: If an unsupported score function is specified.
    """
    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    # Determine the score function based on the input row configuration
    if row['score_func'] == 'chi2':
        score_func = chi2
    elif row['score_func'] == 'f_classif':
        score_func = f_classif
    else:
        raise ValueError("Unsupported score function specified.")

    # Apply SelectKBest to determine the top 'key_feature_number' features
    selector = SelectKBest(score_func, k=key_feature_number)
    selector.fit(p_train_feature, p_train_label)

    # Get the boolean mask or integer index of the features selected
    indexes = selector.get_support()

    # Extract indices of selected features and convert to list
    indices = np.where(indexes)[0].tolist()

    return indices


    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.
    # Get scores for all features
    all_scores = selector.scores_

    # Normalize the scores for all features
    normalized_scores = all_scores / np.sum(all_scores)

    # Extract normalized scores for the selected features
    selected_scores = normalized_scores[indexes]

    # Round the selected normalized scores to 3 decimal places
    rounded_scores = np.round(selected_scores, 3)

    # Sort indices and scores by importance
    sorted_indices_scores = sorted(zip(indices, rounded_scores), key=lambda x: x[1], reverse=True)
    sorted_indices, sorted_scores = zip(*sorted_indices_scores)
    # Calculate accumulated feature importance for selected features
    accumulated_importance = np.round(np.sum(sorted_scores), 3)

    # somtime the score can be 1.01 as we round to 3 decimal we need to check if the sum is more than 1
    if accumulated_importance>1: accumulated_importance=1 

    return list(sorted_indices), list(sorted_scores), accumulated_importance

    """