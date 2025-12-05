import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

# Set seed for reproducibility
import os
import random as rn


def calculate_correlation_edges(X, threshold=0.8):
    """ Generate an edge list based on correlation threshold, skipping constant features. """
    n_features = X.shape[1]
    edge_list = []
    for i in range(n_features):
        if np.std(X[:, i]) == 0:  # Skip if the ith feature is constant
            continue
        for j in range(i + 1, n_features):
            if np.std(X[:, j]) == 0:  # Skip if the jth feature is constant
                continue
            corr, _ = pearsonr(X[:, i], X[:, j])
            if abs(corr) > threshold:
                edge_list.append((i, j))
    return np.array(edge_list)

def calculate_cluster_edges(X, n_clusters):
    """
    Generate an edge list where edges connect features in the same cluster.
    """
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(X.T)  # Transpose to cluster features, not samples
    edge_list = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        for j in cluster_indices:
            for k in cluster_indices:
                if j < k:
                    edge_list.append((j, k))
    return np.array(edge_list)


def soft_threshold(A, b):
    """Apply soft thresholding for lasso regression."""
    res = np.zeros(A.shape)
    res[A > b] = A[A > b] - b
    res[A < -b] = A[A < -b] + b
    return res

def calculate_obj(X, y, w, lambda1, lambda2, T):
    """Calculate the objective function value."""
    return 1/2 * np.linalg.norm(y - np.dot(X, w), 'fro')**2 + lambda1 * np.abs(w).sum() + lambda2 * np.abs(np.dot(T, w)).sum()


def GRAPH_FS(X, y, k, row,  random_state):
    """ Implement the graph structural feature selection algorithm with dynamic edge list determination. """

    """
    Implement the graph structural feature selection algorithm GOSCAR, which applies group lasso with a tree-like
    hierarchical structure to regularize feature selection, aiming to maintain structural integrity among related features.

    Args:
        X (numpy array): Input data matrix with shape (n_samples, n_features).
        y (numpy array): Labels vector with shape (n_samples,).
        row (dict): Dictionary containing all necessary hyperparameters:
            - 'lambda1' (float): Regularization strength for L1 norm of weights.
            - 'lambda2' (float): Regularization strength for hierarchical group lasso penalty.
            - 'rho' (float): Augmented Lagrangian parameter.
            - 'max_iter' (int): Maximum number of iterations for the optimization.
            - 'edge_list' (array): Array defining the group structure as edges between features.
        k (int): Number of features to select, defining the size of the returned feature set.
        seed (int): Random seed for reproducibility in stochastic processes.
        **kwargs:
            - 'verbose' (bool): Flag to enable verbose output during the optimization process.
            - 'method' (str): Method to dynamically generate 'edge_list' if not provided in 'row'. Options are 'correlation' or 'clustering'.
            - 'threshold' (float): Correlation threshold for edge creation if 'method' is 'correlation'.
            - 'n_clusters' (int): Number of clusters for edge creation if 'method' is 'clustering'.

    Returns:
        list: Indices of the top 'k' important features.

    Reference:
        - Liu, Jun, et al. "Moreau-Yosida Regularization for Grouped Tree Structure Learning." NIPS. 2010.
        - Liu, Jun, et al. "SLEP: Sparse Learning with Efficient Projections." http://www.public.asu.edu/~jye02/Software/SLEP, 2009.

    Advice on Setting Hyperparameters:
        - 'lambda1' and 'lambda2' should start from smaller values such as 0.1 or 0.01, especially in high-dimensional settings,
          to prevent excessive regularization that can obscure meaningful feature relationships.
        - 'rho' can be adjusted higher if convergence is slow; however, excessively high values might lead to numerical instability.
        - 'max_iter' may need to be increased in complex datasets or under conditions where convergence is notably sluggish,
          to ensure the algorithm sufficiently explores the solution space.
    """

    # Set seed for reproducibility
    np.random.seed(random_state)
    rn.seed(random_state)

    verbose =  False
    method = 'correlation'  # 'correlation' or 'clustering'
    threshold =  0.8  # Used if method is 'correlation'
    n_clusters = row.get('n_clusters', k)  # Used if method is 'clustering'
    y = y.reshape(-1, 1)  # Ensure y is a column vector

    # Determine the edge list based on the selected method
    if method == 'correlation':
        edge_list = calculate_correlation_edges(X, threshold)
    else:
        edge_list = calculate_cluster_edges(X, n_clusters)

    lambda1 = row['lambda1']
    lambda2 = row['lambda2']
    rho = row['rho']
    max_iter = row['max_iter']
    epsilon = 1e-6  # Start with a small epsilon

    n_samples, n_features = X.shape

    # Construct T from edge_list
    num_edge = edge_list.shape[0]
    T = np.zeros((num_edge * 2, n_features))
    for i in range(num_edge):
        T[i, edge_list[i, 0]] = 1
        T[num_edge + i, edge_list[i, 1]] = -1

    # Attempt to compute the Cholesky decomposition
    F = np.dot(X.T, X) + rho * (np.identity(n_features) + np.dot(T.T, T))

    try:
        R = np.linalg.cholesky(F)
    except np.linalg.LinAlgError:
        # Increase epsilon until the matrix is positive definite
        while True:
            try:
                R = np.linalg.cholesky(F + epsilon * np.identity(n_features))
                break
            except np.linalg.LinAlgError:
                epsilon *= 10  # Increase epsilon
                if verbose:
                    print(f"Increasing epsilon to {epsilon} to achieve positive definiteness.")
                    
    Rinv = np.linalg.inv(R)
    Rtinv = Rinv.T

    # Initialize weights, dual variables
    w = np.zeros((n_features, 1))
    mu = np.zeros_like(w)
    q = np.zeros_like(w)
    p = np.zeros((num_edge * 2, 1))
    v = np.zeros_like(p)

   # Optimization loop
    for iter in range(max_iter):
        b = np.dot(X.T, y) - mu - np.dot(T.T, v) + rho * (np.dot(T.T, p) + q)
        w_hat = np.dot(Rtinv, np.linalg.solve(R, b))
        w = np.dot(Rinv, w_hat)

        q = soft_threshold(w + mu / rho, lambda1 / rho)
        p = soft_threshold(np.dot(T, w) + v / rho, lambda2 / rho)

        mu += rho * (w - q)
        v += rho * (np.dot(T, w) - p)

        if verbose:
            print(f'Objective at iteration {iter}: {calculate_obj(X, y, w, lambda1, lambda2, T)}')

    # Calculate feature importances
    feature_importances = np.abs(w).flatten()
    selected_indices = np.argsort(-feature_importances)[:k]


    return selected_indices
    

    """

    sorted_indices = np.argsort(-normalized_importances)[:k]

    
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.
    # Normalize the importances
    normalized_importances = feature_importances / np.sum(feature_importances)

    # Sort features by normalized importance and limit to top k features
    top_k_normalized_importances = normalized_importances[sorted_indices]

    # Calculate accumulated importance
    accumulated_importance = np.round(np.sum(top_k_normalized_importances), 3)
    if accumulated_importance > 1:
        accumulated_importance = 1  # Adjust if sum exceeds 1 due to rounding
    # Return the results
    return list(sorted_indices), list(np.round(top_k_normalized_importances, 3)), accumulated_importance

    """