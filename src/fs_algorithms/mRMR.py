import functools
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
FLOOR = .001
def groupstats2fstat(avg, var, n):
    """Compute F-statistic of some variables across groups
    Compute F-statistic of many variables, with respect to some groups of instances.
    For each group, the input consists of the simple average, variance and count with respect to each variable.
    Parameters
    ----------
    avg: pandas.DataFrame of shape (n_groups, n_variables)
        Simple average of variables within groups. Each row is a group, each column is a variable.
    var: pandas.DataFrame of shape (n_groups, n_variables)
        Variance of variables within groups. Each row is a group, each column is a variable.
    n: pandas.DataFrame of shape (n_groups, n_variables)
        Count of instances for whom variable is not null. Each row is a group, each column is a variable.
    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic of each variable, based on group statistics.
    Reference
    ---------
    https://en.wikipedia.org/wiki/F-test
    """
    avg_global = (avg * n).sum() / n.sum()  # global average of each variable
    numerator = (n * ((avg - avg_global) ** 2)).sum() / (len(n) - 1)  # between group variability
    denominator = (var * n).sum() / (n.sum() - len(n))  # within group variability
    f = numerator / denominator
    return f.fillna(0.0)
def mrmr_base(K, relevance_func, redundancy_func,
              relevance_args={}, redundancy_args={},
              denominator_func=np.mean, only_same_domain=False,
              return_scores=False, show_progress=True):
    """General function for mRMR algorithm.
    Parameters
    ----------
    K: int
        Maximum number of features to select. The length of the output is *at most* equal to K
    relevance_func: callable
        Function for computing Relevance.
        It must return a pandas.Series containing the relevance (a number between 0 and +Inf)
        for each feature. The index of the Series must consist of feature names.
    redundancy_func: callable
        Function for computing Redundancy.
        It must return a pandas.Series containing the redundancy (a number between -1 and 1,
        but note that negative numbers will be taken in absolute value) of some features (called features)
        with respect to a variable (called target_variable).
        It must have *at least* two parameters: "target_variable" and "features".
        The index of the Series must consist of feature names.
    relevance_args: dict (optional, default={})
        Optional arguments for relevance_func.
    redundancy_args: dict (optional, default={])
        Optional arguments for redundancy_func.
    denominator_func: callable (optional, default=numpy.mean)
        Synthesis function to apply to the denominator of MRMR score.
        It must take an iterable as input and return a scalar.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary redundancy coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """
    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    redundancy = pd.DataFrame(FLOOR, index=features, columns=features)
    K = min(K, len(features))
    selected_features = []
    not_selected_features = features.copy()
    for i in tqdm(range(K), disable=not show_progress):
        score_numerator = relevance.loc[not_selected_features]
        if i > 0:
            last_selected_feature = selected_features[-1]
            if only_same_domain:
                not_selected_features_sub = [c for c in not_selected_features if
                                             c.split('_')[0] == last_selected_feature.split('_')[0]]
            else:
                not_selected_features_sub = not_selected_features
            if not_selected_features_sub:
                redundancy.loc[not_selected_features_sub, last_selected_feature] = redundancy_func(
                    target_column=last_selected_feature,
                    features=not_selected_features_sub,
                    **redundancy_args
                ).fillna(FLOOR).abs().clip(FLOOR)
                score_denominator = redundancy.loc[not_selected_features, selected_features].apply(
                    denominator_func, axis=1).replace(1.0, float('Inf'))
        else:
            score_denominator = pd.Series(1, index=features)
        score = score_numerator / score_denominator
        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)
    if not return_scores:
        return selected_features
    else:
        return (selected_features, relevance, redundancy)

def parallel_df(func, df, series, n_jobs):
    n_jobs = min(cpu_count(), len(df.columns)) if n_jobs == -1 else min(cpu_count(), n_jobs)
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)
def _f_classif(X, y):
    def _f_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]
    return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)
def _f_regression(X, y):
    def _f_regression_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_regression(x[x_not_na].to_frame(), y[x_not_na])[0][0]
    return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)
def f_classif(X, y, n_jobs):
    return parallel_df(_f_classif, X, y, n_jobs=n_jobs)
def f_regression(X, y, n_jobs):
    return parallel_df(_f_regression, X, y, n_jobs=n_jobs)
def _ks_classif(X, y):
    def _ks_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        x = x[x_not_na]
        y = y[x_not_na]
        return x.groupby(y).apply(lambda s: ks_2samp(s, x[y != s.name])[0]).mean()
    return X.apply(lambda col: _ks_classif_series(col, y)).fillna(0.0)
def ks_classif(X, y, n_jobs):
    return parallel_df(_ks_classif, X, y, n_jobs=n_jobs)
def random_forest_classif(X, y):
    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)
def random_forest_regression(X, y):
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)
def correlation(target_column, features, X, n_jobs):
    def _correlation(X, y):
        return X.corrwith(y).fillna(0.0)
    return parallel_df(_correlation, X.loc[:, features], X.loc[:, target_column], n_jobs=n_jobs)
def encode_df(X, y, cat_features, cat_encoding):
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols=cat_features, handle_missing='return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols=cat_features, handle_missing='return_nan'),
        'target': ce.TargetEncoder(cols=cat_features, handle_missing='return_nan')
    }
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    return X
def mrmr_classif(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    """MRMR feature selection for a classification task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "ks" (kolmogorov-smirnov), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.
    n_jobs: int (optional, default=-1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """
    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)
    if relevance == "f":
        relevance_func = functools.partial(f_classif, n_jobs=n_jobs)
    elif relevance == "ks":
        relevance_func = functools.partial(ks_classif, n_jobs=n_jobs)
    elif relevance == "rf":
        relevance_func = random_forest_classif
    else:
        relevance_func = relevance
    redundancy_func = functools.partial(correlation, n_jobs=n_jobs) if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
        np.max if denominator == 'max' else denominator)
    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}
    return mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)
def mrmr_regression(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    """MRMR feature selection for a regression task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.
    n_jobs: int (optional, default=-1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """
    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)
    relevance_func = functools.partial(f_regression, n_jobs=n_jobs) if relevance == 'f' else (
        random_forest_regression if relevance == 'rf' else relevance)
    redundancy_func = functools.partial(correlation, n_jobs=n_jobs) if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
        np.max if denominator == 'max' else denominator)
    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}
    return mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)

import os
import random as rn
import pandas as pd


def mRMR_FS(X, y, K, row, random_state):
    """
    Performs MRMR feature selection followed by classification to determine feature importance.

    Args:
    - X (numpy.ndarray): The feature matrix as a numpy array.
    - y (numpy.ndarray): The target variable as a numpy array.
    - K (int): The number of features to select.
    - row (dict): Dictionary containing hyperparameters for the MRMR feature selection and additional settings.
    - random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        list: Indices of the top 'k' important features.
    """

    # Set the seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)

    # Convert numpy arrays to pandas DataFrame and Series
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')

    # Call MRMR with the unpacked parameters
    selected_features = mrmr_classif(
        X=X_df, 
        y=y_series, 
        K=K,
        relevance=row.get('relevance', 'f'),  # Efficient for linear relationships
        redundancy=row.get('redundancy', 'c'),  # Default Pearson, suitable for initial exploratory feature selection
        denominator=row.get('denominator', 'mean'),  # Normalizes the MRMR score
        cat_features=row.get('cat_features', None),  # Specify if there are categorical features
        cat_encoding=row.get('cat_encoding', 'leave_one_out'),  # Prevent data leakage
        n_jobs=row.get('n_jobs', -1),  # Utilize all cores
        show_progress=row.get('show_progress', False),  # Track progress visually
    )

    # Extract indices from feature names
    selected_feature_indices = [int(name.split('_')[-1]) for name in selected_features]    
    return selected_feature_indices

    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.


    # Train a classifier on the selected features to obtain importances
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X[selected_features], y)

    # Extract and process the importances
    importances = clf.feature_importances_
    normalized_importances = importances / np.sum(importances)
    selected_features = np.argsort(importances)[-K:][::-1]

    selected_normalized_importances = normalized_importances[indices]
    accumulated_importance = np.round(selected_normalized_importances.sum(), 3)
    if accumulated_importance > 1:
        accumulated_importance = 1 

    selected_normalized_importances = np.round(selected_normalized_importances, 3)

    return list(indices), list(np.round(selected_normalized_importances, 3)), accumulated_importance

    """