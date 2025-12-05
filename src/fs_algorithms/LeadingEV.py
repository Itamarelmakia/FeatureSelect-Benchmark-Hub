import numpy as np
import os
import random as rn

def LeadingEV_FS(p_train_feature, k, random_state):
    """
    Performs feature selection based on the Leading Eigenvalues of the covariance matrix.
    This function selects the top 'k' features based on the magnitude of the leading eigenvectors.

    Args:
        p_train_feature (numpy array): Training data with features.
        k (int): Maximum number of top features to select.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        tuple: (sorted_indices, normalized_importances, accumulated_importance)
            - sorted_indices (list): Indices of the top 'k' features selected by the eigenvectors' magnitude.
            - normalized_importances (list): Normalized importance scores of all features, rounded to three decimal places.
            - accumulated_importance (float): Sum of the normalized importance scores of the selected features, indicating total importance captured.
    """
    
    # Set seed for reproducibility
    np.random.seed(random_state)
    rn.seed(random_state)

    # Calculate the covariance matrix
    cov_matrix = np.cov(p_train_feature, rowvar=False)

    """" Another option is to calculate the covariance matrix manually without using np.cov
    # Compute the covariance matrix
    instancesT = p_train_feature.T
    instancesMatrix = np.matmul(instancesT, p_train_feature)
    covEstimator = (1 / rows) * instancesMatrix

    # Transpose the matrix for dot product
    a = covEstimator
    b = a.T

    # Perform dot product without using PyTables
    result = np.dot(a, b)
    """


    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)


    # Get the indices of the k largest eigenvectors
    leadingEV = np.abs(eigenvectors[:, 0])  # Assuming we need the first column
    k = min(len(leadingEV), k)
    LeadingEVIndexes = np.argpartition(leadingEV, -k)[-k:]
    indices_for_selected = LeadingEVIndexes.tolist()  



    return indices_for_selected
    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    
    # Compute the importance of features based on the sum of absolute values of eigenvectors (column-wise sum)
    feature_importances = np.sum(np.abs(eigenvectors), axis=1)

    # Normalize the importance scores
    normalized_importances = feature_importances / np.sum(feature_importances)
    sorted_indices = np.argsort(-normalized_importances)[:k]
    top_k_normalized_importances = normalized_importances[sorted_indices]

    # Calculate accumulated importance
    accumulated_importance = np.round(np.sum(top_k_normalized_importances), 3)
    if accumulated_importance > 1:
        accumulated_importance = 1  # Adjust if sum exceeds 1 due to rounding

    return list(sorted_indices), list(np.round(top_k_normalized_importances, 3)), accumulated_importance

    """
    