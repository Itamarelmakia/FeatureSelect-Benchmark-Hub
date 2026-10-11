import numpy as np
import os
import random as rn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def LASSO_Stability_FS(X_train, y_train, k, row, random_state):
    """
    Stability selection via bootstrap + L1-penalised Logistic Regression.

    For each of B bootstrap half-samples, fit L1-LR and record which features
    have non-zero coefficients. Return the top-k features by selection frequency.

    Args:
        X_train: (n_samples, n_features) float array
        y_train: (n_samples,) integer class labels
        k: number of features to return
        row: hyperparameters with keys 'n_bootstrap', 'selection_threshold', 'C'
        random_state: int seed

    Returns:
        list of k feature indices (0-indexed)
    """
    seed = int(random_state)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    X = np.array(X_train, dtype=float)
    y = np.array(y_train).ravel()
    n, p = X.shape

    n_bootstrap = int(row['n_bootstrap'])
    C = float(row['C'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rng = np.random.RandomState(seed)
    selection_counts = np.zeros(p, dtype=float)

    for b in range(n_bootstrap):
        half = n // 2
        idx = rng.choice(n, size=half, replace=False)
        X_b = X_scaled[idx]
        y_b = y[idx]

        try:
            lr = LogisticRegression(
                penalty='l1',
                C=C,
                solver='liblinear',
                max_iter=500,
                random_state=seed + b,
                multi_class='ovr',
            )
            lr.fit(X_b, y_b)
            coef = lr.coef_
            selected = np.any(np.abs(coef) > 1e-8, axis=0)
            selection_counts += selected.astype(float)
        except Exception:
            pass

    # selection probability
    sel_prob = selection_counts / n_bootstrap

    # top-k by selection frequency
    top_k = np.argsort(sel_prob)[::-1][:k]
    return list(top_k)
