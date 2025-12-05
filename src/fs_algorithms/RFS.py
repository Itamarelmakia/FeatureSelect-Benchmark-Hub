import numpy as np
import random as rn
from numpy import linalg as LA

# Set seed for reproducibility
import os
import random as rn


def generate_diagonal_matrix(U):
    """
    This function generates a diagonal matrix D from an input matrix U as D_ii = 0.5 / ||U[i,:]||

    Input:
    -----
    U: {numpy array}, shape (n_samples, n_features)

    Output:
    ------
    D: {numpy array}, shape (n_samples, n_samples)
    """
    temp = np.sqrt(np.multiply(U, U).sum(1))
    temp[temp < 1e-16] = 1e-16
    temp = 0.5 / temp
    D = np.diag(temp)
    return D

def calculate_l21_norm(X):
    """
    This function calculates the l21 norm of a matrix X, i.e., \sum ||X[i,:]||_2

    Input:
    -----
    X: {numpy array}, shape (n_samples, n_features)

    Output:
    ------
    l21_norm: {float}
    """
    return (np.sqrt(np.multiply(X, X).sum(1))).sum()





def one_hot_encode_labels(Y):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # This sets the output as a dense matrix directly
    Y_one_hot = encoder.fit_transform(Y)  # Ensure Y is the correct shape
    return Y_one_hot


def calculate_obj(X, Y, W, gamma):
    """
    This function calculates the objective function of RFS.
    """
    temp = np.dot(X, W) - Y
    return calculate_l21_norm(temp) + gamma * calculate_l21_norm(W)


def RFS_FS(X, Y,  k,row,random_state ):

    """
    This function implements efficient and robust feature selection via joint l21-norms minimization
    min_W||X^T W - Y||_2,1 + gamma||W||_2,1

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data.
    Y: {numpy array}, shape (n_samples, n_classes)
        Input class label matrix, each row is a one-hot-coding class label.
    row: {dict}
        Dictionary containing hyperparameters, including:
        gamma: {float}
            Regularization parameter for RFS (smaller gamma recommended for high-dimensional datasets).
        max_iter: {int}
            Maximum number of iterations for the RFS algorithm (default is 1000).
    k: {int}
        Number of top features to select.
    verbose: {boolean}
        True to display the objective function value at each iteration, False otherwise.

    Returns:
        list: Indices of the top 'k' important features.

    Key Modifications:
    - Added row to input hyperparameters 'gamma' and 'max_iter'.
    - Added k to select the top k features based on weights.
    - For high-dimensional datasets (where #rows << #columns), a small gamma value (e.g., gamma = 0.01) is recommended to capture important features.
   
    Reference
    ---------
    Nie, Feiping et al. "Efficient and Robust Feature Selection via Joint l2,1-Norms Minimization" NIPS 2010.
    """

    verbose=False

    # Set seed for reproducibility
    np.random.seed(random_state)
    rn.seed(random_state)

    gamma = row['gamma']  # Regularization parameter
    max_iter = row.get('max_iter', 1000)

    n_samples, n_features = X.shape
    A = np.zeros((n_samples, n_samples + n_features))
    A[:, :n_features] = X
    A[:, n_features:] = gamma * np.eye(n_samples)
    D = np.eye(n_features + n_samples)

    Y = Y.reshape(-1, 1)  # Reshape Y to (-1, 1) because fit_transform expects 2D array

    # Update Y to be one-hot encoded
    Y = one_hot_encode_labels(Y) # Y_one_hot


    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        D_inv = np.linalg.inv(D)
        temp = np.linalg.inv(A @ D_inv @ A.T + 1e-6 * np.eye(n_samples))
        U = D_inv @ A.T @ temp @ Y

        #print("Shape of U:", U.shape)  # Debug statement to check the shape of U
        #print("Shape of D_inv:", D_inv.shape)  # Debug statement to check the shape of D_inv
        #print("Shape of temp:", temp.shape)  # Debug statement to check the shape of temp
        #print("Shape of Y:", Y.shape)  # Debug statement to check the shape of Y


        if U.ndim == 1:
            U = U[:, np.newaxis]  # Ensure U is two-dimensional

        D = generate_diagonal_matrix(U)

        obj[iter_step] = calculate_obj(X, Y, U[:n_features, :], gamma)

        if verbose:
            print(f'Objective at iteration {iter_step + 1}: {obj[iter_step]}')
        if iter_step > 0 and abs(obj[iter_step] - obj[iter_step - 1]) < 1e-3:
            break

    W = U[:X.shape[1], :]
    feature_importance = np.linalg.norm(W, axis=1)

    top_k_indices = np.argsort(feature_importance)[-k:][::-1]
    top_k_feature_importance = feature_importance[top_k_indices]

    return top_k_indices

"""
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.
    
    # Sort features by normalized importance and limit to top k features
    sorted_indices = np.argsort(-normalized_importances)[:k]
    top_k_normalized_importances = normalized_importances[sorted_indices]

    # Calculate accumulated importance
    accumulated_importance = np.round(np.sum(top_k_normalized_importances), 3)
    if accumulated_importance > 1:
        accumulated_importance = 1  # Adjust if sum exceeds 1 due to rounding

    # Return the results
    return list(sorted_indices), list(np.round(top_k_normalized_importances, 3)), accumulated_importance

"""
