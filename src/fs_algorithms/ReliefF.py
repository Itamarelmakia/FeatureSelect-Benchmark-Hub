# pip install skrebate - to install the skrebate package
import skrebate
from skrebate import ReliefF
import numpy as np
import random as rn
import os


def ReliefF_FS(train_data, train_labels, k, row, num_classes, random_state):
    """
    Perform feature selection using the ReliefF algorithm.

    Args:
        train_data (numpy array): Training data with features.
        train_labels (numpy array): Labels corresponding to the training data.
        k (int): Number of top features to select.
        row (dict): Dictionary containing hyperparameters, including:
            n_neighbors (int): Number of neighbors to use for ReliefF.
        num_classes (int): Number of unique classes in the target variable.
        random_state (int): Seed used by the random number generator for reproducibility.

    Returns:
        list: Indices of the top 'k' important features.


    Reference
    ---------
    Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.

    
    """
    
    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    
    max_n_neighbors = num_classes  # Define the maximum number of neighbors
    n_neighbors = min(max_n_neighbors, row.get('n_neighbors', k))  # Number of neighbors


    # Check the data type of train_data
    if train_data.dtype == np.uint8:
        print("Converting train_data from uint8 to float64 to avoid division errors.")
        train_data = train_data.astype(np.float64)
        
    # Initialize the ReliefF algorithm with the specified number of neighbors
    reliefF = ReliefF(n_neighbors=n_neighbors)

    train_data = train_data.astype('float64')
    # Fit the ReliefF algorithm to the training data
    reliefF.fit(train_data, train_labels)

    # Get feature importances
    feature_importances = reliefF.feature_importances_


    # Get the indices of the top 'k' features
    indices = np.argsort(feature_importances)[-k:]

    return indices

    # Normalize the importances for all features
    normalized_importances = importances / np.sum(importances)

    # Sort features by normalized importance
    indices = np.argsort(-normalized_importances)

    # Select and sort top k features
    selected_features = indices[:k]


    return selected_features

    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    top_k_normalized_importances = normalized_importances[top_k_indices]

    # Calculate accumulated importance for the top k features
    accumulated_importance = np.round(np.sum(top_k_normalized_importances), 3)
    if accumulated_importance > 1:
        accumulated_importance = 1  # Correcting in case of rounding issues

    return list(top_k_indices), list(np.round(top_k_normalized_importances, 3)), accumulated_importance
    """
