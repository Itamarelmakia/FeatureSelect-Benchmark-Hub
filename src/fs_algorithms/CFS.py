import os
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
import scipy.io
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from itertools import compress



import os
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime
import warnings
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


import os
import numpy as np
import pandas as pd
import random as rn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import entropy
import scipy.io

# Set seed for reproducibility
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
rn.seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def su_calculation(f1, f2):
    """
    Calculate symmetrical uncertainty between two features, optimizing the use of entropy and mutual information.
    """
    joint_entropy = entropy(np.vstack((f1, f2)), base=2)
    entropy_f1 = entropy(f1, base=2)
    entropy_f2 = entropy(f2, base=2)
    mutual_info = entropy_f1 + entropy_f2 - joint_entropy
    return 2 * mutual_info / (entropy_f1 + entropy_f2) if (entropy_f1 + entropy_f2) > 0 else 0

def merit_calculation(X, y):
    """
    Efficiently calculates the merit of X given class labels y.
    """
    n_features = X.shape[1]
    rff = 0
    rcf = 0
    for i in range(n_features):
        rcf += su_calculation(X[:, i], y)
        for j in range(i + 1, n_features):
            rff += su_calculation(X[:, i], X[:, j])
    rff *= 2
    return rcf / np.sqrt(n_features + rff)

def cfs(X, y, k,row,seed):
    """
    Optimized CFS algorithm that selects up to k features based on their merits.
    """
    n_features = X.shape[1]
    selected_features = []
    feature_scores = []

    while len(selected_features) < k:
        best_feature = None
        best_merit = float('-inf')
        for i in range(n_features):
            if i not in selected_features:
                current_set = selected_features + [i]
                current_merit = merit_calculation(X[:, current_set], y)
                if current_merit > best_merit[]:
                    best_merit = current_merit
                    best_feature = i
        if best_feature is None:
            break
        selected_features.append(best_feature)
        feature_scores.append(best_merit)
        # Early stopping condition if no new features improve the merit
        if len(feature_scores) > 5 and all(x <= feature_scores[-1] for x in feature_scores[-5:]):
            break

    return selected_features, feature_scores

