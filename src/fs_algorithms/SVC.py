import os
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime, timedelta
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from itertools import compress

def SVC_FS(p_train_feature, p_train_label, k, row, random_state):
    """
    Perform feature selection using a Support Vector Classifier (SVC). This function selects the top 'k' features
    based on the weights of attributes in the SVC model.

    Args:
        p_train_feature (numpy array): Training data with features.
        p_train_label (numpy array): Labels corresponding to the training data.
        k (int): Maximum number of top features to select.
        row (dict): Configuration dictionary with parameters for the SVC:
            - 'C' (float): Regularization parameter.
            - 'penalty' (str): Specifies the norm used in the penalization ('l1' or 'l2').
            - 'dual' (bool): Dual or primal formulation. Prefer dual=False when n_samples > n_features.
        random_state (int): Seed used by the random number generator for reproducibility.

    Returns:
        list: Indices of the top 'k' important features.

    Example:
        indices, importance, total_importance = SVC_FS(train_data, train_labels, 10, 
                                                      {'C': 0.01, 'penalty': 'l1', 'dual': False}, 42)
    """

    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    lsvc = LinearSVC(C=row['C'], penalty=row['penalty'],dual=row['dual'], random_state=random_state).fit(p_train_feature, p_train_label)
    model = SelectFromModel(lsvc, prefit=True,max_features=k)
    indexes = model.get_support()
    selected_features = list(compress(range(len(indexes)), indexes))
    
    return selected_features

    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    # Calculate accumulated importance
    accumulated_importance = np.round(normalized_importances[:k].sum(), 3)

    # somtime the score can be 1.01 as we round to 3 decimal we need to check if the sum is more than 1
    if accumulated_importance>1: accumulated_importance=1 

    return list(top_k_indices), list(normalized_importances[:k]), accumulated_importance

    """