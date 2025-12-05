import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import random as rn
import os

def ETree_FS(train_data, train_labels, k, row, random_state):
    """
    Select features using the Extra Trees classifier. This function returns the indices of the top 'k' important features
    based on their importance scores, along with their normalized importance values and the accumulated importance.

    Args:
        train_data (numpy array): Training data with features.
        train_labels (numpy array): Labels corresponding to the training data.
        k (int): Number of top features to select.
        row (dict): Dictionary containing hyperparameters for the classifier:
            - 'n_estimators' (int): The number of trees in the forest.
            - 'max_depth' (int or None): The maximum depth of the trees.
        random_state (int): Seed used by the random number generator for reproducibility.

    Returns:
        list: Indices of the top 'k' important features.

    """

    # Set the seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    # Initialize the classifier with hyperparameters
    clf = ExtraTreesClassifier(n_estimators=row['n_estimators'], max_depth=row['max_depth'], random_state=random_state)
    
    # Fit the model
    clf.fit(train_data, train_labels)
    
    # Get feature importances from the model
    importances = clf.feature_importances_
    


    # Get indices of the top 'k' important features in descending order
    selected_features = np.argsort(importances)[-k:][::-1]

    return selected_features


    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    # Normalize the importances
    normalized_importances = importances / np.sum(importances)

    # Extract normalized scores for the selected features
    selected_normalized_importances = normalized_importances[indices]

    # Accumulated importance of the normalized scores
    accumulated_importance = np.round(selected_normalized_importances.sum(), 3)

    # Sometimes the score can be 1.01 as we round to 3 decimal places; need to check if the sum is more than 1
    if accumulated_importance > 1:
        accumulated_importance = 1 

    return list(indices), list(np.round(selected_normalized_importances, 3)), accumulated_importance

    """

