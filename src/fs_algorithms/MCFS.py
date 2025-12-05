import scipy
import numpy as np
from sklearn import linear_model

import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
import os
import random as rn

def MCFS_FS(X, num_classess, k, row,random_state, **kwargs):

    """
    Perform unsupervised feature selection for multi-cluster data, returning sorted indices and their normalized importances.

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data
    k: {int}
        Number of features to select
    kwargs: {dictionary}
        W: {sparse matrix}, shape (n_samples, n_samples) - affinity matrix
        n_clusters: {int} - number of clusters, default is 5, if you have Y label use the number of classes as k

    Output
    ------
    tuple: (sorted_indices, normalized_importances, accumulated_importance)
        sorted_indices (list): Indices of the top 'k' features selected by the model.
        normalized_importances (list): Normalized importance scores of the selected features.
        accumulated_importance (float): Sum of the normalized importance scores.


    Reference
    ---------
    Cai, Deng et al. "Unsupervised Feature Selection for Multi-Cluster Data." KDD 2010.
    """

    from utilities import construct_W
    import scipy
    import numpy as np
    from sklearn import linear_model
    from scipy.sparse import csc_matrix
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    import random as rn
    from datetime import datetime
    import os
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Suppress the ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning) # in many High dimentional dataset with sparse dataset we didn't converge - I want to avoid form print the 
    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)



    # Building the affinity matrix with dynamic parameters
    kwargs_W = {
        "metric": row['metric'],
        "neighbor_mode": row['neighbor_mode'],
        "weight_mode": row['weight_mode'],
        "k": num_classess,  # using the number of unique labels as 'k' if applicable
        "t": X.shape[1],  # use feature count as the 't' parameter in heat kernel
        "fisher_score": row.get('fisher_score', False),
        "reliefF": row.get('reliefF', False)
    }
    W = construct_W(X, **kwargs_W)



    # Set default number of clusters if not provided
    if 'n_clusters' not in kwargs:
        n_clusters = num_classess
    else:
        n_clusters = kwargs['n_clusters']

    # solve the generalized eigen-decomposition problem and get the top K
    # eigen-vectors with respect to the smallest eigenvalues
    W = W.toarray()
    W = (W + W.T) / 2
    W_norm = np.diag(np.sqrt(1 / W.sum(1)))
    W = np.dot(W_norm, np.dot(W, W_norm))
    WT = W.T
    W[W < WT] = WT[W < WT]
    eigen_value, ul = scipy.linalg.eigh(a=W)
    Y = np.dot(W_norm, ul[:, -1*n_clusters-1:-1])

    # solve K L1-regularized regression problem using LARs algorithm with cardinality constraint being d
    n_sample, n_feature = X.shape
    W = np.zeros((n_feature, n_clusters))
    for i in range(n_clusters):
        clf = linear_model.Lars(n_nonzero_coefs=k)
        clf.fit(X, Y[:, i])
        W[:, i] = clf.coef_
    
    idx = feature_ranking(W,k)
    len(idx)
    return idx


def feature_ranking(W,n_selected_features):
    """
    This function computes MCFS score and ranking features according to feature weights matrix W
    
    Parameters:
    W (numpy.ndarray): Feature weights matrix.
    n_selected_features (int): Number of top features to select.
    
    Returns:
    numpy.ndarray: Indices of the top n_selected_features features.
    """
    # Compute MCFS score
    mcfs_score = W.max(1)

    # Rank features based on MCFS score
    idx = np.argsort(mcfs_score, 0)
    idx = idx[::-1]

    # Return the top n_selected_features features
    return idx[:n_selected_features]


    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    sorted_scores = normalized_importances[sorted_indices]

    # Compute accumulated importance
    accumulated_importance = np.round(np.sum(sorted_scores), 3)
    if accumulated_importance > 1:  # Correct if sum exceeds 1 due to rounding
        accumulated_importance = 1

    return list(sorted_indices), list(sorted_scores), accumulated_importance


    """
