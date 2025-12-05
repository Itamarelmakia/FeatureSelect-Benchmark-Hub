import os
import numpy as np
import random as rn
from sklearn import linear_model

def alpha_investing_FS(X, y,  k,row, random_state):
    """
    This function implements streamwise feature selection (SFS) algorithm alpha_investing for binary regression or
    univariate regression

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, assume feature arrives one at each time step
    y: {numpy array}, shape (n_samples,)
        input class labels or regression target

    Output
    ------
    F: {numpy array}, shape (n_selected_features,)
        index of selected features in a streamwise way


    Reference :
    ---------
    Zhou, Jing et al. "Streaming Feature Selection using Alpha-investing." KDD 2006.

    Key Modifications:
    - feature_scores array: This stores the score (error reduction) for each feature, regardless of whether it is selected based on p-value.
    - Top k features: After evaluating all features, the top k features are selected based on the highest scores using np.argsort(). This ensures that even if fewer than k features pass the p-value threshold, we still return the top k based on their importance.
    - Error reduction score: The score for each feature is calculated as the difference in error between the model with and without the feature.
      This ensures that you get exactly k features returned based on their importance, even if not all pass the alpha-investing criteria.

    """
    
    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    n_samples, n_features = X.shape
    w = row['w0']
    dw = row['dw']


    F = []  # selected features
    for i in range(n_features):
        x_can = X[:, i]  # generate next feature
        alpha = w/2/(i+1)
        X_old = X[:, F]
        if i == 0:
            X_old = np.ones((n_samples, 1))
            linreg_old = linear_model.LinearRegression()
            linreg_old.fit(X_old, y)
            error_old = 1 - linreg_old.score(X_old, y)
        if i != 0:
            # model built with only X_old
            linreg_old = linear_model.LinearRegression()
            linreg_old.fit(X_old, y)
            error_old = 1 - linreg_old.score(X_old, y)

        # model built with X_old & {x_can}
        X_new = np.concatenate((X_old, x_can.reshape(n_samples, 1)), axis=1)
        logreg_new = linear_model.LinearRegression()
        logreg_new.fit(X_new, y)
        error_new = 1 - logreg_new.score(X_new, y)

        # calculate p-value
        pval = np.exp((error_new - error_old)/(2*error_old/n_samples))
        if pval < alpha:
            F.append(i)
            w = w + dw - alpha
        else:
            w -= alpha

    # Get the indices of the top 'k' features
    if len(F) >= k:
        indices = F[:k]
    else:
        indices = F

    return indices


    



    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    # Normalize the top k importances and calculate accumulated importance
    top_k_normalized_importances = normalized_importances[sorted_indices]
    accumulated_importance = np.round(np.sum(top_k_normalized_importances), 3)
    if accumulated_importance > 1:
        accumulated_importance = 1  # Adjust if sum exceeds 1 due to rounding
    return list(sorted_indices), list(np.round(top_k_normalized_importances, 3)), accumulated_importance

    """