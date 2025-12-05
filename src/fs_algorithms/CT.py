import numpy as np
import pandas as pd
import os
import random as rn

def CT_FS(X_train, k, row,random_state):
    """
    Perform covariance thresholding feature selection. This method uses a thresholded covariance matrix to determine
    the principal components and selects features based on the leading eigenvalues' contribution.

    Args:
        X_train (numpy array): Input feature matrix with shape (n_samples, n_features).
        k (int): Number of top features to select.
        row (dict): Dictionary containing hyperparameters for the thresholding:
            - 't' (float): Threshold value for covariance values.

    Returns:
        list: Indices of the top 'k' important features.
          
    Explanation:
        - The function first computes the covariance matrix from the provided data array.
        - It then applies a threshold to zero out elements below the threshold while preserving the diagonal.
        - Eigenvalues and eigenvectors of the thresholded covariance matrix are calculated.
        - The top 'k' features are selected based on the magnitude of the leading eigenvectors.
        - Feature importance is calculated as the sum of squared loadings for each feature, normalized, and sorted.
    """


    # Set the seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    rows, columns = X_train.shape
    t = row['t'] / (rows ** 0.5)
    instancesT = X_train.T
    instancesMatrix = np.matmul(instancesT, X_train)
    covEstimator = (1 / rows) * (instancesMatrix)
    sparsedCovEstimator = np.copy(covEstimator)
    originalDiagonal = sparsedCovEstimator.diagonal()
    
    # Thresholding the covariance matrix
    sparsedCovEstimator[sparsedCovEstimator < t] = 0
    np.fill_diagonal(sparsedCovEstimator, originalDiagonal)
    
    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(sparsedCovEstimator)
    
    # Sort eigenvectors by the magnitude of eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]
    sorted_eigenvalues = eigenvalues[idx]

    # Select the top 'k' features based on the leading eigenvalues
    leadingEVSparsed = sorted_eigenvectors[:, :k]
    selected_features = np.argpartition(abs(leadingEVSparsed).sum(axis=1), -k)[-k:]


    # Calculate feature importance with RandomForest after feature selection
    #sorted_indices, sorted_importance_rounded, accumulated_importance = calculate_feature_importance_rf(X_train, y_train, selected_features)
    
    return selected_features



"""
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.


    # Calculate feature importance based on the sum of squared loadings of each feature
    feature_importance_all = np.square(leadingEVSparsed).sum(axis=1)

    # Normalize the feature importance by the sum of all feature importances
    normalized_importance_all = feature_importance_all / feature_importance_all.sum()

    # Extract feature importance for the selected features
    selected_feature_importance = normalized_importance_all[indices_for_selected]

    # Sort by importance
    sorted_indices_importance = sorted(zip(indices_for_selected, selected_feature_importance), key=lambda x: x[1], reverse=True)
    sorted_indices, sorted_importance = zip(*sorted_indices_importance)

    # Filter only the first k features and round to 3 decimal places
    top_k_indices = sorted_indices[:k]
    top_k_importances = sorted_importance[:k]
    top_k_importances_rounded = np.round(top_k_importances, 3)

    # Calculate accumulated importance
    accumulated_importance = np.round(np.sum(top_k_importances_rounded), 3)
    if accumulated_importance > 1:
        accumulated_importance = 1  # Adjust if sum exceeds 1 due to rounding

    # Return the results
    return list(top_k_indices), list(top_k_importances_rounded), accumulated_importance


"""