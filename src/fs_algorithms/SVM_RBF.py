from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import os, random as rn

def SVM_RBF_FS(X_train, y_train, k, row, random_state):
    """
    Feature selection with an RBF-kernel SVM using permutation importance.

    Args
    ----
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels (1-D).
    k       : int
        Number of top features to return.
    row     : dict
        Hyper-parameters for the SVM:
            'C'     (float, default=1.0)
            'gamma' (str|float, default='scale')
            'n_repeats' (int, permutation repeats, default=10)
    random_state : int
        Seed for full reproducibility.

    Returns
    -------
    tuple (indices, normalized_importances, accumulated_importance)
        indices                – numpy array of selected column indices
        normalized_importances – relative importances (sum = 1)
        accumulated_importance – float, ==1.0
    """

    # deterministic behaviour
    seed = random_state
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    # hyper-params with defaults
    C          = row.get("C", 1.0)
    gamma      = row.get("gamma", "scale")
    n_repeats  = row.get("n_repeats", 10)

    # SVM pipeline (standardisation + RBF SVM)
    svm = SVC(kernel="rbf", C=C, gamma=gamma, random_state=random_state)
    pipe = Pipeline([("sc", StandardScaler()), ("svc", svm)])

    # fit on all features
    pipe.fit(X_train, y_train)

    # permutation importance
    perm = permutation_importance(
        pipe, X_train, y_train,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    importances = perm.importances_mean          # array of length n_features

    # top-k indices
    indices = np.argsort(importances)[-k:]

    # normalise
    sel_scores = importances[indices]
    normalized = sel_scores / sel_scores.sum()
    accumulated_importance = normalized.sum()    # == 1.0

    return indices#, normalized, accumulated_importance