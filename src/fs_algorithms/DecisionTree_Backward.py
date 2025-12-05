import numpy as np
import random as rn
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning) # in many High dimentional dataset with sparse dataset we didn't converge - I want to avoid form print the 


def initialize_decision_tree(X_train, row, random_state):
    num_columns = X_train.shape[1]
    if num_columns > 10000:
        max_features = int(np.sqrt(num_columns))
        max_leaf_nodes = 100
        splitter = 'random'
    elif num_columns > 1000:
        max_features = int(np.sqrt(num_columns))
        max_leaf_nodes = 500
        splitter = 'random'
    elif num_columns > 100:
        max_features = int(np.sqrt(num_columns))
        max_leaf_nodes = 1000
        splitter = 'best'
    else:
        max_features = None
        max_leaf_nodes = None
        splitter = 'best'

    return DecisionTreeClassifier(
        max_depth=row.get('max_depth'),
        min_samples_split=row.get('min_samples_split', 2),
        min_samples_leaf=row.get('min_samples_leaf', 1),
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        splitter=splitter,
        class_weight=row.get('class_weight', 'balanced'),
        random_state=random_state
    )

def DecisionTree_Backward_FS(X_train, y_train, k, row, random_state):


    """
    Performs backward feature selection using a decision tree. It iteratively removes the feature
    that contributes the least to model accuracy until only 'k' features remain. Then, it uses
    a RandomForest to assess the importance of the selected features.

    Args:
    - X_train (numpy.ndarray): The training feature matrix.
    - y_train (numpy.ndarray): The training labels.
    - k (int): The target number of features to retain.
    - row (dict): Dictionary containing hyperparameters.
    - random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
    - tuple: (list of selected feature indices, list of normalized importance scores, float accumulated importance)
    """

    # Set the seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    n_features = X_train.shape[1]
    selected_features = list(range(n_features))

    while len(selected_features) > k:
        worst_feature = None
        best_score = float('-inf')

        for feature in selected_features:
            trial_features = [f for f in selected_features if f != feature]
            X_train_subset = X_train[:, trial_features]

            dtc = initialize_decision_tree(X_train_subset, row, random_state)
            dtc.fit(X_train_subset, y_train)
            score = accuracy_score(y_train, dtc.predict(X_train_subset))

            if score > best_score:
                best_score = score
                worst_feature = feature

        if worst_feature is not None:
            selected_features.remove(worst_feature)
    return selected_features



    """
    # Calculate feature importance with ExtraTreesClassifier after feature selection
    import sys
    sys.path.append(utilites_path)
    from utilities import calculate_feature_importance_rf

    # Calculate feature importance with RandomForest after feature selection
    sorted_indices, sorted_importance_rounded, accumulated_importance = calculate_feature_importance_rf(X_train, y_train, selected_features)
    return sorted_indices, sorted_importance_rounded, accumulated_importance
    """