import numpy as np
from sklearn.feature_selection import VarianceThreshold
import random as rn
import os

def low_variance_FS(X, k, row, random_state):
    """
    Perform feature selection based on low variance. This method selects the top 'k' features that have the highest
    variance across the dataset, considering a variance threshold to filter out features with lower variance first.

    Args:
        X (numpy array): Input data matrix with shape (n_samples, n_features).
        k (int): Number of top features to select.
        row (dict): Configuration dictionary with parameters:
            - 'p' (float): Proportion threshold to set variance threshold as p * (1 - p).

    Returns:
        list: Indices of the top 'k' features sorted by importance.


    Explanation of Choosing p=0.1 for High-Dimensional Sparse Data :
        Understanding the Variance Threshold:

        The variance threshold is calculated as p * (1 - p). For p=0.1, the threshold becomes 0.1 * (1 - 0.1) = 0.1 * 0.9 = 0.09.
        This threshold is used to filter out features with low variance, which are less likely to be informative.
        High-Dimensional Sparse Data:

        High-dimensional datasets have a large number of features (dimensions) compared to the number of samples.
        Sparse data means that many of the features have zero or near-zero values for most samples.
        Why p=0.1 is Appropriate:

        Filtering Out Low-Variance Features: In high-dimensional sparse datasets, many features may have very low variance because they contain mostly zeros. Setting p=0.1 helps to filter out these low-variance features, retaining only those with higher variance that are more likely to be informative.
        Balancing Feature Selection: A lower p value (e.g., p=0.01) would set a lower threshold, potentially retaining too many low-variance features. A higher p value (e.g., p=0.5) would set a higher threshold, potentially filtering out too many features. p=0.1 strikes a balance by setting a moderate threshold.
        Reducing Dimensionality: By filtering out low-variance features, you reduce the dimensionality of the dataset, which can improve the performance of machine learning algorithms and reduce computational complexity.
    """
    
    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    
    # Extract the proportion threshold 'p' from the configuration dictionary
    p = row['p']
    # Calculate the variance threshold as p * (1 - p)
    threshold = p * (1 - p)
    # Initialize the VarianceThreshold object with the calculated threshold
    sel = VarianceThreshold(threshold)

    try:
        # Apply the variance threshold to filter out low variance features
        X_new = sel.fit_transform(X)
        if X_new.shape[1] == 0:
            # If no features meet the variance threshold, raise a ValueError
            raise ValueError(f"No feature meets the variance threshold of {threshold}.")
    except ValueError:
        # If no features meet the initial variance threshold, adjust the threshold and try again
        p /= 10
        threshold = p * (1 - p)
        sel = VarianceThreshold(threshold)
        X_new = sel.fit_transform(X)
        if X_new.shape[1] == 0:
            # If still no feature meets the variance threshold, proceed without it
            sel = VarianceThreshold(0)  # This effectively removes the variance threshold
            X_new = sel.fit_transform(X)

    # Get the variances of the features
    variances = sel.variances_
    # Sort the indices of the features by their variances in descending order
    sorted_indices = np.argsort(variances)[::-1]
    # Select the top 'k' features with the highest variances
    top_k_indices = sorted_indices[:k]
    # Get the variances of the top 'k' features
    top_k_variances = variances[top_k_indices]

    return top_k_indices

    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    # Normalize the variances for all features
    normalized_variances = variances / np.sum(variances)
    top_k_normalized_variances = normalized_variances[top_k_indices]

    # Calculate the accumulated importance of the normalized scores
    accumulated_importance = np.round(np.sum(top_k_normalized_variances), 3)
    # Ensure that the accumulated importance does not exceed 1
    if accumulated_importance > 1:
        accumulated_importance = 1.0

    return list(top_k_indices), list(np.round(top_k_normalized_variances, 3)), accumulated_importance

    """
