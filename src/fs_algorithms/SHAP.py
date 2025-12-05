from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import shap, numpy as np, os, random as rn

def SHAP_FS(X_train, y_train, k, row, random_state):
    """
    Select k important features using mean |SHAP| values.

    • Binary → GradientBoostingClassifier  (as you had before)
    • Multi-class → RandomForestClassifier (SHAP supports it natively)
    """
     # 1 ─ reproducibility ------------------------------------------------
    seed = random_state
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); rn.seed(seed)


    # 2 ─ hyper-parameters ----------------------------------------------
    n_estimators  = row.get("n_estimators" , 100)
    learning_rate = row.get("learning_rate", 0.1)
    max_depth     = row.get("max_depth"    , 3)


    # 3 ─ choose a model SHAP supports ----------------------------------
    if len(np.unique(y_train)) == 2:                    # binary
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed
        )
    else:                                               # ≥ 3 classes
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=seed
        )


    model.fit(X_train, y_train)



    # 4 ─ SHAP values ----------------------------------------------------
    explainer = shap.TreeExplainer(model, X_train)
    try:
        shap_vals = explainer.shap_values(X_train)
    except shap.utils._exceptions.ExplainerError:
        # If additivity check fails, disable it and try again
        shap_vals = explainer.shap_values(X_train, check_additivity=False)  # disables additivity check if needed


    if isinstance(shap_vals, list):                     # multi-class
        shap_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
    else:                                               # binary
        shap_abs = np.abs(shap_vals)


    # 5 ─ global importance and TOP-k -----------------------------------
    scores  = shap_abs.mean(axis=0)                     # one score / feature
    k       = min(k, X_train.shape[1])
    indices = np.argsort(scores)[-k:][::-1]             # descending order


    return indices.ravel().astype(int)   # ← **guaranteed shape (k,)**




