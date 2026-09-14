# config.py – Simplified for Real World Datasets only

import importlib
import os
import pandas as pd 
import sys

# Debug mode flag - set to True to enable detailed printouts during processing
debug=False

# Fixed settings (using baseline assumptions only)
pilot_list_dataset = 'y'
use_baseline_assumptions = 'y'
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)

# Baseline parameters (only one experiment type is supported)
K_List =  [1,5,10,30,60] # ,89
Experimnet_Type = '5CV'
initial_splits = 5
final_splits = 2
HighT = 0.9
LowT = 0.7
perform_cv = 'Y'

# FS algorithm list
fs_algorithms_list = 'All'  # All / OR choose from list

# define predict thrhsold
fixed_threshold = 0.5

train_size = 1 / initial_splits if perform_cv.lower() == 'y' else 0.5


evaluation_models = ["ETREE", "LR", "SVM", "RF", "KNN","LDA"]


# Paths setup


# Get the directory of this config.py file
config_dir = os.path.dirname(os.path.abspath(__file__))
# Go up to the project root (parent of src)
project_root = os.path.abspath(os.path.join(config_dir, '..', '..'))
# Now set data_path to the Real World Datasets folder at the project root
data_path = os.path.join(project_root, 'data')


path_output_excel_FS = os.path.join(project_root, 'results/FS/')
#path_output_excel_MCI = os.path.join(project_root, 'results/Human Aging/')
#convert_data_hard_path_Source = os.path.join(project_root, 'results/Convert Data 2 Hard/')
pilot_name = "FS_Real_World_Dataset"
# Ensure required directories exist
paths_to_check = [config_dir, data_path, path_output_excel_FS]#, convert_data_hard_path_Source, path_output_excel_MCI]
for folder_path in paths_to_check:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Hyperparameters file path
Loadpath_hyper = os.path.join(config_dir, 'Algorithms_Hyperparameters_FS.xlsx')
Loadpath_hyper = Loadpath_hyper.replace('\\', '/')



def update_config(specific_experiment, perform_cv_value, path_output_excel_MCI_value):
    """
    Since we are only running Real World Datasets, we always use the default splits.
    """
    global initial_splits, train_size, path_output_excel_FS
    initial_splits = 5  # default for Real World Datasets
    if perform_cv_value.lower() == 'y':
        train_size = 1 / initial_splits
    else:
        train_size = 0.5


src_dir = os.path.abspath(os.path.join(config_dir, '..'))
fs_algorithms_dir = os.path.join(src_dir, 'fs_algorithms')
sys.path.append(fs_algorithms_dir)
sys.path.append(os.path.join(fs_algorithms_dir, 'DLFS'))
sys.path.append(os.path.join(fs_algorithms_dir, 'CAE'))
sys.path.append(os.path.join(fs_algorithms_dir, 'UFS'))

# ---------------------------------------------------------------------------
# Feature selection algorithm registry
#
# Each selector is imported defensively. The neural methods (CAE, DLFS, UFS,
# GRACES) depend on optional heavy packages (tensorflow, torch,
# torch_geometric, keras, h5py) that deliberately are not part of the core
# `requirements.txt`. A selector whose dependencies are missing is recorded in
# `unavailable_algorithms` and left out of `feature_selection_mapping`, instead
# of aborting the entire benchmark at import time.
#
# When every dependency IS installed, the resulting mapping is identical to the
# original explicit import list -- the benchmark semantics are unchanged.
# ---------------------------------------------------------------------------

# (algorithm name, module to import, function to pull out of that module)
ALGORITHM_IMPORTS = [
    ('ETree', 'ETree', 'ETree_FS'),
    ('AdaBoost', 'AdaBoost', 'AdaBoost_FS'),
    ('Univariate', 'Univariate', 'Univariate_FS'),
    ('CT', 'CT', 'CT_FS'),
    ('low_variance', 'low_variance', 'low_variance_FS'),
    ('SVC', 'SVC', 'SVC_FS'),
    ('SVM_RBF', 'SVM_RBF', 'SVM_RBF_FS'),
    ('SHAP', 'SHAP', 'SHAP_FS'),
    ('LS', 'LS', 'LS_FS'),
    ('Skewness', 'Skewness', 'Skewness_FS'),
    ('MCFS', 'MCFS', 'MCFS_FS'),
    ('DT', 'DT', 'DT_FS'),
    ('ReliefF', 'ReliefF', 'ReliefF_FS'),
    ('alpha_investing', 'alpha_investing', 'alpha_investing_FS'),
    ('DecisionTree_Forward', 'DecisionTree_Forward', 'DecisionTree_Forward_FS'),
    ('SVM_Forward', 'SVM_Forward', 'SVM_Forward_FS'),
    ('GRAPH', 'GRAPH', 'GRAPH_FS'),
    ('RFS', 'RFS', 'RFS_FS'),
    ('LeadingEV', 'LeadingEV', 'LeadingEV_FS'),
    ('mRMR', 'mRMR', 'mRMR_FS'),
    ('GRACES', 'GRACES', 'GRACES_FS'),
    ('DecisionTree_Backward', 'DecisionTree_Backward', 'DecisionTree_Backward_FS'),
    ('SVM_Backward', 'SVM_Backward', 'SVM_Backward_FS'),
    ('CEI_GA', 'CEI_GA', 'CEI_GA_FS'),
    # Implemented in their own sub-folders (already added to sys.path above)
    ('DLFS', 'DLFS', 'DLFS_FS'),
    ('CAE', 'CAE', 'CAE_FS'),
    ('UFS', 'UFS', 'UFS_FS'),
    # 'CARTE' is implemented in fs_algorithms/CARTE.py but is not part of the
    # published benchmark and is intentionally left unregistered.
]

# Mapping of algorithm names to their feature selection functions.
feature_selection_mapping = {}
# Algorithm name -> reason it could not be loaded (missing optional dependency).
unavailable_algorithms = {}

for _algo_name, _module_name, _function_name in ALGORITHM_IMPORTS:
    try:
        _module = importlib.import_module(_module_name)
        feature_selection_mapping[_algo_name] = getattr(_module, _function_name)
    except Exception as _import_error:  # noqa: BLE001 - report, never abort
        unavailable_algorithms[_algo_name] = (
            f"{type(_import_error).__name__}: {_import_error}"
        )


def describe_algorithm_availability():
    """Human readable summary of which selectors this environment can run."""
    lines = [
        f"Feature selection algorithms available: "
        f"{len(feature_selection_mapping)}/{len(ALGORITHM_IMPORTS)}"
    ]
    if unavailable_algorithms:
        lines.append(
            "Unavailable (optional dependency missing - install the extras in "
            "requirements-optional.txt to enable them):"
        )
        for name in sorted(unavailable_algorithms):
            lines.append(f"  - {name}: {unavailable_algorithms[name]}")
    return "\n".join(lines)


# Define hyperparameter variables
UnivariateFS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='Univariate', engine='openpyxl')
low_variance_hyper = pd.read_excel(Loadpath_hyper, sheet_name='low_variance', engine='openpyxl')
ETree_hyper = pd.read_excel(Loadpath_hyper, sheet_name='ETree', engine='openpyxl')
Skewness_hyper = pd.read_excel(Loadpath_hyper, sheet_name='Skewness', engine='openpyxl')

SVM_RBF_hyper =  pd.read_excel(Loadpath_hyper, sheet_name='SVM_RBF', engine='openpyxl')
SHAP_hyper =  pd.read_excel(Loadpath_hyper, sheet_name='SHAP', engine='openpyxl')

SVC_hyper = pd.read_excel(Loadpath_hyper, sheet_name='SVC', engine='openpyxl')
LS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='LS', engine='openpyxl')
MCFS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='MCFS', engine='openpyxl')
mRMR_hyper = pd.read_excel(Loadpath_hyper, sheet_name='mRMR', engine='openpyxl')
CT_hyper = pd.read_excel(Loadpath_hyper, sheet_name='CT', engine='openpyxl')
AdaBoost_hyper = pd.read_excel(Loadpath_hyper, sheet_name='AdaBoost', engine='openpyxl')
ReliefFFS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='ReliefF', engine='openpyxl')
UFS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='UFS', engine='openpyxl')
DLFS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='DLFS', engine='openpyxl')
CAE_hyper = pd.read_excel(Loadpath_hyper, sheet_name='CAE', engine='openpyxl')

SVM_Forward_FS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='SVM_Forward', engine='openpyxl')
DecisionTree_Backward_hyper = pd.read_excel(Loadpath_hyper, sheet_name='DecisionTree_Backward', engine='openpyxl')
GRACES_FS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='GRACES', engine='openpyxl')
alpha_investing_hyper = pd.read_excel(Loadpath_hyper, sheet_name='alpha_investing', engine='openpyxl')
RFS_hyper = pd.read_excel(Loadpath_hyper, sheet_name='RFS', engine='openpyxl')
GRAPH_Backward_hyper = pd.read_excel(Loadpath_hyper, sheet_name='GRAPH', engine='openpyxl')
SVM_Backward_hyper = pd.read_excel(Loadpath_hyper, sheet_name='SVM_Backward', engine='openpyxl')

CEI_GA_hyper = pd.read_excel(Loadpath_hyper, sheet_name='CEI_GA', engine='openpyxl')





# CAE have hyperparamters but we use the developers hyperparameters
no_hyperparameters = ['DT','LeadingEV']
# NN Algorithms
algo_name_NN = ['CAE', 'UFS', 'DLFS', 'GRACES']

# `feature_selection_mapping` is built above from ALGORITHM_IMPORTS.


