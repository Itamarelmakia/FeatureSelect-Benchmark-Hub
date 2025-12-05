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

def DecisionTree_Forward_FS(X_train, y_train, k, row, random_state):
    """
    Performs forward feature selection using a decision tree. It iteratively adds the feature
    that contributes the most to model accuracy until 'k' features are selected. Then, it uses
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
    selected_features = []
    remaining_features = list(range(n_features))

    while len(selected_features) < k:
        best_feature = None
        best_score = float('-inf')

        for feature in remaining_features:
            trial_features = selected_features + [feature]
            X_train_subset = X_train[:, trial_features]

            dtc = initialize_decision_tree(X_train_subset, row, random_state)
            dtc.fit(X_train_subset, y_train)
            score = accuracy_score(y_train, dtc.predict(X_train_subset))

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)


    return selected_features