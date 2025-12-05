import numpy as np
import os
import random as rn
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os
import random as rn

import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning) # in many High dimentional dataset with sparse dataset we didn't converge - I want to avoid form print the 


def SVM_Forward_FS(X_train, y_train, k, row, random_state):
    """
    Performs forward feature selection using LinearSVC, selecting features that improve model accuracy
    and returning the indices of selected features along with their normalized importance scores and accumulated importance.

    Args:
        X_train (numpy.ndarray): The training feature matrix.
        y_train (numpy.ndarray): The training labels.
        k (int): The target number of features to retain.
        row (dict): Dictionary containing hyperparameters.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        tuple: (list of selected feature indices, list of normalized importance scores, float accumulated importance)
    """
    # Import the calculate_feature_importance_rf function from utilities
    from utilities import calculate_feature_importance_rf

    np.random.seed(random_state)
    rn.seed(random_state)

    n_features = X_train.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))

    while len(selected_features) < k:
        best_feature = None
        best_score = float('-inf')

        for feature in remaining_features:
            trial_features = selected_features + [feature]
            X_train_subset = X_train[:, trial_features]

            svc = LinearSVC(max_iter=row.get('max_iter', 10000), random_state=random_state)
            svc.fit(X_train_subset, y_train)
            score = accuracy_score(y_train, svc.predict(X_train_subset))

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

    # Calculate feature importance with RandomForest after feature selection
    import sys
    sys.path.append(utilites_path)

    sorted_indices, sorted_importance_rounded, accumulated_importance = calculate_feature_importance_rf(X_train, y_train, selected_features,random_state)
    return sorted_indices, sorted_importance_rounded, accumulated_importance


import numpy as np
import os
import random as rn
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning) # in many High dimentional dataset with sparse dataset we didn't converge - I want to avoid form print the 


def SVM_Forward_FS(X_train, y_train, k, row, random_state):
    """
    Performs forward feature selection using LinearSVC, selecting features that improve model accuracy
    and returning the indices of selected features along with their normalized importance scores and accumulated importance.

    Args:
        X_train (numpy.ndarray): The training feature matrix.
        y_train (numpy.ndarray): The training labels.
        k (int): The target number of features to retain.
        row (dict): Dictionary containing hyperparameters.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        tuple: (list of selected feature indices, list of normalized importance scores, float accumulated importance)
    """


    # Set the seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    n_features = X_train.shape[1]

    # Determine max_iter based on the number of features
    if n_features > 10000:
        max_iter = 250
    elif n_features > 1000:
        max_iter = 500
    elif n_features > 100:
        max_iter = 1000
    else:
        max_iter = 10000

    selected_features = []
    remaining_features = list(range(n_features))

    while len(selected_features) < k:
        best_feature = None
        best_score = float('-inf')

        for feature in remaining_features:
            trial_features = selected_features + [feature]
            X_train_subset = X_train[:, trial_features]

            svc = LinearSVC(max_iter=max_iter, random_state=random_state)
            svc.fit(X_train_subset, y_train)
            score = accuracy_score(y_train, svc.predict(X_train_subset))

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)



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