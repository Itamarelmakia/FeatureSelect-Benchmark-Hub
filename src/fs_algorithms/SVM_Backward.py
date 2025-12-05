
import numpy as np
import random as rn
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
import random as rn

import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning) # in many High dimentional dataset with sparse dataset we didn't converge - I want to avoid form print the 


def SVM_Backward_FS(X_train, y_train, k, row, random_state):
    """
    Performs backward feature selection using LinearSVC, starting with all features and iteratively removing features
    that offer the least contribution to the model accuracy until 'k' features are left. After selection, a RandomForestClassifier
    is used to compute the importance of the selected features.

    Args:
    - X_train (numpy.ndarray): The training feature matrix.
    - y_train (numpy.ndarray): The training labels.
    - k (int): The number of features to keep.
    - row (dict): Dictionary containing hyperparameters for LinearSVC.
    - random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
    - tuple: (selected_indices, normalized_importances, accumulated_importance)
        - selected_indices (list): Indices of the selected features of size 'k'.
        - normalized_importances (list): Normalized importance scores of the selected features.
        - accumulated_importance (float): Total importance captured by the selected features.
    """

    # Set the seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    n_features = X_train.shape[1]
    selected_features = list(range(n_features))

    # Determine max_iter based on the number of features
    if n_features > 10000:
        max_iter = 250
    elif n_features > 1000:
        max_iter = 500
    elif n_features > 100:
        max_iter = 1000
    else:
        max_iter = 10000

    while len(selected_features) > k:
        best_subset = None
        best_score = float('-inf')

        for feature in selected_features:
            trial_features = [f for f in selected_features if f != feature]
            trial_train_feature = X_train[:, trial_features]

            lsvc = LinearSVC(C=row['C'], penalty=row['penalty'], dual=row.get('dual', False), tol=row['tol'], max_iter=max_iter, random_state=random_state)
            lsvc.fit(trial_train_feature, y_train)
            score = accuracy_score(y_train, lsvc.predict(trial_train_feature))

            if score > best_score:
                best_score = score
                best_subset = trial_features

        selected_features = best_subset if best_subset is not None else selected_features
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