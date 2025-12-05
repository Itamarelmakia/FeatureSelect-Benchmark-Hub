import numpy as np
import random as rn
import os


def DT_FS(p_train_feature, k, random_state):
    """
    Perform feature selection based on diagonal thresholding of the covariance matrix.
    This function selects the top 'k' features based on the magnitude of the diagonal values of the covariance matrix.

    Args:
        p_train_feature (numpy array): Training data with features.
        k (int): Maximum number of top features to select.
        random_state (int): Seed for random number generation to ensure reproducibility.

    Returns:
        list: Indices of the top 'k' features selected by the diagonal values.
    """
    
    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    # Get the number of rows and columns in the training data
    rows, columns = p_train_feature.shape

    # Transpose the training data
    instancesT = p_train_feature.T

    # Compute the covariance matrix
    instancesMatrix = np.matmul(instancesT, p_train_feature)
    covEstimator = (1 / rows) * instancesMatrix

    # Extract the diagonal values of the covariance matrix
    diagonal = covEstimator.diagonal()

    # Ensure k does not exceed the number of features
    k = min(len(diagonal), k)

    # Get the indices of the top 'k' features based on the diagonal values
    DiagonalIndexes = np.argpartition(diagonal, -k)[-k:]
    indices_for_selected = DiagonalIndexes.tolist()

    return indices_for_selected

    




    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    # Calculate accumulated importance of the selected features
    accumulated_importance = np.round(np.sum(rounded_importances[top_k_indices]), 3)
    if accumulated_importance > 1:  # Correcting potential rounding error causing the sum to exceed 1
        accumulated_importance = 1.0
    return list(top_k_indices), list(rounded_importances[top_k_indices]), accumulated_importance
    """