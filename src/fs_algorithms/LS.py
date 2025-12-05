

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
import os
import random as rn

def LS_FS(p_train_feature, num_classes, k, row, random_state):
    """
    Main function to compute the Laplacian Score for feature selection using an affinity matrix and ranks features by their score.

    Args:
        p_train_feature (numpy.ndarray): Training feature data.
        num_classes (int): Number of classes (used here for determining some matrix dimensions, typically 'k').
        k (int): Number of top features to select.
        row (dict): Dictionary of parameters for constructing the affinity matrix and additional configuration.
        seed (int): Seed for reproducibility.

    Returns:
        list: Indices of the top 'k' selected features sorted by importance,
              Normalized scores of these features rounded to 3 decimal places,
              Accumulated importance of these features rounded to 3 decimal places.
    """
    from utilities import construct_W  # Ensure this utility function is correctly implemented

    def lap_score(X, W):
        """
        Calculate Laplacian scores for each feature based on the provided affinity matrix.

        Args:
            X (numpy.ndarray): Input data matrix.
            W (scipy.sparse.csc_matrix): Affinity matrix.

        Returns:
            numpy.ndarray: Array of scores for each feature.
        """
        D = np.array(W.sum(axis=1)).flatten()
        D = scipy.sparse.diags(D)
        L = D - W  # Laplacian matrix
        Xt = X.T
        f = Xt.dot(D.dot(X)) - Xt.dot(W.dot(X))
        f = np.diagonal(f)
        d = Xt.dot(D.dot(np.ones(X.shape[0])))
        scores = f / d
        return scores
    

    

    def feature_ranking(score):
        """
        Rank features based on the Laplacian score and sort them by importance.
        """
        # Sorting the features by score in descending order
        sorted_indices = np.argsort(-score)
        return sorted_indices, score[sorted_indices]

    # Set the seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    # Building the affinity matrix with dynamic parameters
    kwargs_W = {
        "metric": row['metric'],
        "neighbor_mode": row['neighbor_mode'],
        "weight_mode": row['weight_mode'],
        "k": num_classes,  # using the number of unique labels as 'k' if applicable
        "t": p_train_feature.shape[1],  # use feature count as the 't' parameter in heat kernel
        "fisher_score": row.get('fisher_score', False),
        "reliefF": row.get('reliefF', False)
    }
    W = construct_W(p_train_feature, **kwargs_W)


    # Calculating Laplacian Scores
    scores = lap_score(p_train_feature, W)

    # Ranking and selecting the top 'k' features
    sorted_indices, sorted_scores = feature_ranking(scores)

    # Normalize the scores for all features
    #normalized_scores = np.round(sorted_scores / np.sum(sorted_scores), 3)
    # Selecting top 'k' indices 
    top_k_indices = sorted_indices[:k]

    return top_k_indices
    
    
    
    """ 
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    # Selecting top 'k' indices and their normalized scores
    top_k_indices = sorted_indices[:k]
    top_k_normalized_scores = normalized_scores[:k]

    # Calculate accumulated importance from the top 'k' normalized scores
    accumulated_importance = np.round(np.sum(top_k_normalized_scores), 3)

    # Sometimes the score can be 1.01 as we round to 3 decimal places; need to check if the sum is more than 1
    if accumulated_importance > 1:
        accumulated_importance = 1

    return list(top_k_indices), list(top_k_normalized_scores), accumulated_importance

    """
   