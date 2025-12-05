import numpy as np
import pandas as pd
from scipy.stats import skew
import operator
import time
import random as rn
import os

def Skewness_FS(data,  k,row,random_state):
    """
    Perform feature selection based on the skewness of features, evaluating their effectiveness using an SVM classifier.

    Args:
        data (pd.DataFrame): Input data with features.
        y (array-like): Target variable.
        k (int): Number of top features to select.

    Returns:
        pd.DataFrame: Results containing the number of features selected, time taken, and classification accuracy.
    """

    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    # Convert X to a pandas DataFrame
    X_df = pd.DataFrame(data)

    # Calculate skewness for each feature with variance check
    def calculate_skewness(x):
        if np.var(x) == 0:
            return 0
        return skew(x.dropna(), nan_policy='omit') 
        """
          Avoid from error:  "/cnvrg/Algorithms/Skewness.py:72: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
         skewness_scores = X_df.apply(lambda x: skew(x.dropna(), nan_policy='omit'))"""
    
    # Calculate skewness for each feature
    skewness_scores = X_df.apply(calculate_skewness)


    if (  len(X_df.columns)/len(X_df)) > 2 :
        # Filter features by skewness threshold
        filtered_features = skewness_scores[np.abs(skewness_scores) > row['skew_threshold']]
        if len(filtered_features) <k :
           filtered_features = skewness_scores
    else :
        # Filter features by skewness threshold
        filtered_features = skewness_scores[np.abs(skewness_scores) > row['skew_threshold']/2]
        if len(filtered_features) <k :
           filtered_features = skewness_scores

        # Sort features by absolute skewness in descending order and select top k features
    top_features = filtered_features.abs().sort_values(ascending=False).head(k)
    top_k_indices = top_features.index.tolist()
    top_k_skewness = top_features.values.tolist()

    return top_k_indices

    """

    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    top_k_skewness = sorted_normalized_scores[:k]

    # Round the normalized scores
    rounded_normalized_scores = np.round(top_k_skewness, 3)

    # Calculate accumulated importance
    accumulated_importance = np.round(rounded_normalized_scores.sum(), 3)
    if accumulated_importance > 1:
        accumulated_importance = 1  # Correct for any potential rounding error above 1

  
    return list(top_k_indices), list(rounded_normalized_scores.iloc[:k]), accumulated_importance
    """