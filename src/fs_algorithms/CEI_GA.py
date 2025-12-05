# ------------------------------------------------------------------
# CEI_GA.py
#   Classification Error Impurity (CEI) feature selector
# ------------------------------------------------------------------
#  CEI is a frequency-based filter ranker (Nematzadeh et al., 2024)
#  that measures how “pure” each feature’s value-ordering is w.r.t.
#  the class label.
#
#  Steps for a single feature
#  --------------------------
#  1. Sort the samples by that feature → reorder the label vector Y.
#  2. For each class c:
#       • Take the contiguous window from c’s first to last appearance.
#       • Let  n_other   = #labels in that window ≠ c
#         and n_current = #labels in that window  = c
#       • Compute w_c  = n_other / (n_other + n_current)                (Eq. 4.1)
#  3. CEI = 1 – min_c w_c                                              (Alg. 1, line 10)
#     → perfect separability ⇒ min_c w_c = 0 ⇒ CEI = 1 (best score).
#
#  The selector returns the k highest-scoring features together with
#  their CEI scores normalised to sum to 1.
# ------------------------------------------------------------------

import numpy as np, os, random as rn
import numpy as np, os, random as rn
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import random 
# --------------------------------------------------------------------
#               CEI + GA   (author’s CMF-AGAwER pipeline)
# --------------------------------------------------------------------
def CEI_GA_FS(X, y, k, row, random_state):
    """
    Two-stage feature selector:
    1. Filter  – CEI ranking, keep pre_keep best (default 50).
    2. Wrapper – GA with β-radius mutation, choose final subset.

    Parameters
    ----------
    X, y : ndarray
        Training data and labels.
    k    : int
        Upper bound on features to keep (overridden by row['max_features']).
    row  : dict – optional keys
        • 'max_features' : int   – final subset target (default = k)
        • 'pre_keep'     : int   – CEI cut-off q  (default = 50)
        • 'ga_beta'      : int   – GA β radius   (default = 2)
        • 'ga_pop'       : int   – population size (default = 20)
        • 'ga_gen'       : int   – # generations  (default = 50)
        • 'cv_splits'    : int   – folds in fitness (default = 5)
    random_state : int

    Returns
    -------
    indices                – list of selected column indices
    normalised_importances – CEI scores of those indices (sum = 1)
    accumulated_importance – 1.0
    """

    # ---------------- reproducibility --------------------------------
    seed = random_state
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); rn.seed(seed)

    # ---------------- hyper-params -----------------------------------
    max_features = k  #int(row.get('max_features', k))
    pre_keep     = int(row.get('pre_keep', 50))               # paper’s q
    ga_beta      = int(row.get('ga_beta', 2))                 # paper’s β
    ga_pop       = int(row.get('ga_pop', 20))
    ga_gen       = int(row.get('ga_gen', 50))
    cv_splits    = int(row.get('cv_splits', 5))

    # ---------------- CEI ranking (parameter-free) -------------------
    n_samples, n_features = X.shape
    cei_scores = np.empty(n_features, dtype=float)

    for j in range(n_features):
        order  = np.argsort(X[:, j], kind="mergesort")
        labels = y[order]
        w_vals = []
        for cls in np.unique(labels):
            first, last = np.where(labels == cls)[0][[0, -1]]
            window      = labels[first:last + 1]
            n_other     = (window != cls).sum()
            n_current   = (window == cls).sum()
            w_vals.append(n_other / (n_other + n_current))
        cei_scores[j] = 1.0 - min(w_vals)        # CEI = 1 – min w_c

    # keep the top-q features
    q = min(pre_keep, n_features)
    cei_idx = np.argsort(cei_scores)[-q:]
    cei_subset_scores = cei_scores[cei_idx]

    # ----------- GA wrapper (binary mask over the q features) ---------
    rng = random.Random(seed)

    def fitness(mask):
        cols = [f for use, f in zip(mask, cei_idx) if use]
        if not cols:              # avoid empty subset
            return 0.0
        X_sub = X[:, cols]
        acc   = 0.0
        cv = StratifiedKFold(cv_splits, shuffle=True, random_state=seed)
        for tr, te in cv.split(X_sub, y):
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X_sub[tr], y[tr])
            acc += clf.score(X_sub[te], y[te]) / cv_splits
        return acc

    n_bits = q
    pop = [[rng.choice([0, 1]) for _ in range(n_bits)] for _ in range(ga_pop)]
    scores = [fitness(ind) for ind in pop]

    for _ in range(ga_gen):
        # tournament-2 selection
        parents = [max(rng.sample(list(zip(pop, scores)), 2),
                       key=lambda t: t[1])[0] for _ in range(ga_pop)]

        # mutation with β-radius
        new_pop = []
        for p in parents:
            child = p[:]
            for i in range(n_bits):
                if rng.random() < 1 / (ga_beta * n_bits):
                    child[i] ^= 1
            new_pop.append(child)

        pop = new_pop
        scores = [fitness(ind) for ind in pop]

    best_mask = pop[int(np.argmax(scores))]
    chosen = [f for use, f in zip(best_mask, cei_idx) if use]

    # honour overall max_features cap
    chosen = chosen[-max_features:] if len(chosen) > max_features else chosen

    # ------------- importance rescaling for return -------------------
    final_scores = cei_scores[chosen]
    norm_imp = final_scores / final_scores.sum() if final_scores.sum() \
               else np.full_like(final_scores, 1.0 / len(final_scores))

    return chosen # , norm_imp, 1.0