# main.py # Ver 1/1/25
# 1) Silence the low‑level INFO log:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # only WARNING or above

# 2) Import TF and configure threading for CPU work:
import tensorflow as tf
# e.g. limit each op to 4 threads, and run up to 2 ops in parallel
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# 3) Configure GPU memory growth so you don’t pre‑allocate everything:
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


import torch

# Operating System and System-Specific
import os
import sys
import os.path


# Data Manipulation and Visualization
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import scipy.io
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq

# Parallel Processing
import time

import multiprocessing as mp

# force every child to be a fresh Python process (no forked CUDA context)
mp.set_start_method('spawn', force=True)


from joblib import Parallel, delayed

# File Handling
import glob
import openpyxl
import xlrd 
import xlsxwriter
import re
import random as rn

# Miscellaneous
import copy
import warnings



from configs.config import *

# list of FS methods that can use the GPU
GPU_ALGOS = ['CAE', 'UFS', 'DLFS', 'GRACES']




# Append directories to sys.path using os.path.join.
path = os.getcwd()
sys.path.append(path + '/fs_algorithms')      
sys.path.append(path + '/fs_algorithms/DLFS')      
sys.path.append(path + '/fs_algorithms/CAE')     
sys.path.append(path + '/fs_algorithms/UFS')      

# Import algorithm functions.
from fs_algorithms.ETree import ETree_FS
from fs_algorithms.AdaBoost import AdaBoost_FS
from fs_algorithms.Univariate import Univariate_FS
from fs_algorithms.CT import CT_FS
from fs_algorithms.SHAP import SHAP_FS
from fs_algorithms.SVM_RBF import SVM_RBF_FS


from fs_algorithms.low_variance import low_variance_FS
from fs_algorithms.SVC import SVC_FS
from fs_algorithms.LS import LS_FS
from fs_algorithms.Skewness import Skewness_FS
from fs_algorithms.MCFS import MCFS_FS
from fs_algorithms.DT import DT_FS
from fs_algorithms.ReliefF import ReliefF_FS
from fs_algorithms.alpha_investing import alpha_investing_FS
from fs_algorithms.DecisionTree_Forward import DecisionTree_Forward_FS
from fs_algorithms.SVM_Forward import SVM_Forward_FS
from fs_algorithms.GRAPH import GRAPH_FS
from fs_algorithms.RFS import RFS_FS
from fs_algorithms.LeadingEV import LeadingEV_FS
from fs_algorithms.mRMR import mRMR_FS
from fs_algorithms.GRACES import GRACES_FS
from fs_algorithms.DecisionTree_Backward import DecisionTree_Backward_FS
from fs_algorithms.SVM_Backward import SVM_Backward_FS
from fs_algorithms.CEI_GA import CEI_GA_FS




# Try to import from a single file; if that fails, import from the subfolder.
try:
    # Import from single file if all functions are stored in one place
    from fs_algorithms.DLFS import DLFS_FS
    from fs_algorithms.CAE import CAE_FS
    from fs_algorithms.UFS import UFS_FS
except ImportError:
    # Import from folder if functions are stored in multiple files within a folder
    from fs_algorithms.DLFS.DLFS import DLFS_FS
    from fs_algorithms.CAE.CAE import CAE_FS
    from fs_algorithms.UFS.UFS import UFS_FS


# In utilities.py

def get_algorithms_mapping():
    """
    Combines the mapping of algorithm names to their corresponding feature selection functions
    and hyperparameters. Returns a dictionary with algorithm names as keys and a dictionary with
    keys 'function' and 'hyper' as values.
    """


    # Mapping of algorithm names to their hyperparameters (read from Excel or set as string)
    hyperparameters_mapping = {
        'Univariate': UnivariateFS_hyper,
        'low_variance': low_variance_hyper,
        'ETree': ETree_hyper,
        'SVC': SVC_hyper,
        'SVM_RBF': SVM_RBF_hyper,
        'SHAP' : SHAP_hyper,
        'LS': LS_hyper,
        'Skewness': Skewness_hyper,
        'MCFS': MCFS_hyper,
        'DT': 'No_hyperparamters',
        'AdaBoost': AdaBoost_hyper,
        'ReliefF': ReliefFFS_hyper,
        'alpha_investing': alpha_investing_hyper,
        'DecisionTree_Forward': DecisionTree_Backward_hyper,
        'DLFS': DLFS_hyper,
        'SVM_Forward': SVM_Forward_FS_hyper,
        'GRAPH': GRAPH_Backward_hyper,
        'RFS': RFS_hyper,
        'LeadingEV': 'No_hyperparamters',
        'CAE': CAE_hyper,
        'mRMR': mRMR_hyper,
        'CT': CT_hyper,
        'GRACES': GRACES_FS_hyper,
        'UFS': UFS_hyper,
        'DecisionTree_Backward': DecisionTree_Backward_hyper,
        'SVM_Backward': SVM_Backward_hyper,
        'CEI_GA': CEI_GA_hyper
    }

    algorithms = {}
    for algo_name, func in feature_selection_mapping.items():
        hyper = hyperparameters_mapping.get(algo_name, None)
        algorithms[algo_name] = {'function': func, 'hyper': hyper}
    return algorithms


def get_sorted_algorithms(Alg_Group):
    """
    Returns a sorted list of algorithm mapping dictionaries (with keys: 'name', 'function', 'hyper')
    based on a predetermined order.
    """
    
    sorted_order_All = [
         'ETree','AdaBoost','Univariate','low_variance', 'SVC','LS','MCFS','DT', 'Skewness', 'alpha_investing', 'SHAP',
         'ReliefF','mRMR','CT','LeadingEV',  'RFS',
         'SVM_Forward', 'CEI_GA','SVM_RBF',
         
         'DecisionTree_Forward', 'DecisionTree_Backward', 'SVM_Backward',
         
          'DLFS','GRAPH',   'CAE', 'GRACES','UFS'
         
    ]

    sorted_order_Fast = [
         'ETree','AdaBoost','Univariate','low_variance', 'SVC','LS','MCFS','DT', 'Skewness', 'alpha_investing', 'SHAP',
         'ReliefF','mRMR','CT','LeadingEV',  'RFS',
         'SVM_RBF','SVM_Forward', 'CEI_GA',
    ]
    sorted_order_GPUs = [
         'DLFS','CAE', 'GRACES','SVM_RBF','UFS'
         
    ]

    sorted_order_Slow = [
         'GRAPH', 'RFS','DecisionTree_Backward', 'SVM_Backward'
    ]
    if Alg_Group == 'All' :
        sorted_order =sorted_order_All
    else :
        if Alg_Group == 'Fast' :
            sorted_order =sorted_order_Fast
        else :
            if Alg_Group == 'Slow' :
                sorted_order =sorted_order_Slow
            else :
                if Alg_Group == 'GPU' :
                    sorted_order =sorted_order_GPUs

    mapping = get_algorithms_mapping()
    sorted_algos = []
    for algo in sorted_order:
        if algo in mapping:
            sorted_algos.append({
                'name': algo,
                'function': mapping[algo]['function'],
                'hyper': mapping[algo]['hyper']
            })
    return sorted_algos

def read_parquet_file(file_path):
    """
    Read a Parquet file and return a DataFrame.
    """
    return pd.read_parquet(file_path)

def get_shape(df):
    """
    Get the number of rows and columns in a DataFrame.
    """
    return df.shape

def get_unique_folder_names(file_path):
    folder_names = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f)) and f != ".ipynb_checkpoints"]
    unique_folder_names = list(set(folder_names))
    return unique_folder_names

def get_file_names_without_extension(data_path):
    mat_files = glob.glob(os.path.join(data_path, '*.mat'))
    file_names_without_extension = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]
    return file_names_without_extension

def remove_mat_extension(string):
    if string.endswith(".mat"):
        return string[:-4]
    else:
        return string

def dataset_hardness_score(df, HighT, LowT):
    """
    Calculate the dataset hardness score based on accuracy and AUC.

    This function performs the following steps:
    1. Define conditions for accuracy and AUC based on the provided thresholds.
    2. Assign complexity scores ('High', 'Low', 'Ok') based on the conditions.
    3. Add new columns 'FS_Dataset_Complexity_Score_Acc' and 'FS_Dataset_Complexity_Score_AUC' to the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the necessary columns.
    HighT (float): The high threshold for accuracy and AUC.
    LowT (float): The low threshold for accuracy and AUC.

    Returns:
    pd.DataFrame: The updated DataFrame with the new columns added.
    """

    # Define conditions for accuracy
    conditions_acc = [
        df['Average Accuracy'] >= HighT,
        df['Average Accuracy'] < LowT
    ]
    # Define choices corresponding to the conditions for accuracy
    choices_acc = ['High', 'Low']

    # Define conditions for AUC
    conditions_auc = [
        df['Average AUC'] >= HighT,
        df['Average AUC'] < LowT
    ]
    # Define choices corresponding to the conditions for AUC
    choices_auc = ['High', 'Low']

    # Default value if none of the conditions are met
    default = 'Ok'

    # Apply conditions and choices to create new columns
    df['FS_Dataset_Complexity_Score_Acc'] = np.select(conditions_acc, choices_acc, default=default)
    df['FS_Dataset_Complexity_Score_AUC'] = np.select(conditions_auc, choices_auc, default=default)

    return df



def read_or_create_result_table(path_output_excel_FS,specific_experiment):
    """
    Read or create a new result table with a fixed set of columns.

    Args:
        path_output_excel_FS (str): The path to the output Excel directory.
        specific_experiment (str): The name of the pilot (used to name the output file).

    Returns:
        tuple: A tuple containing the result DataFrame and a list of its columns.
    """
    import os
    import pandas as pd

    # Build the full output file path.
    pathresults = os.path.join(path_output_excel_FS, f"{specific_experiment}.xlsx")

    # Define the required columns in the specified order.
    columns = [
        "Data_Algo_rows_columns_k_Hyperparamter_Threshold_Evaluator_Key",
        "Repository",
        "Dataset",
        "FS/HD",
        "Algorithm",
        "N", "P",  # Number of rows and number of features.
        "P/N",
        "P/N Classification",
        "K",
        'K_%',
        'Ln(K_%)',
        "Hyperparamter",
        "Perform_CV[Y/N]",
        "#Folds",
        "#Successfuls_folds",
        "Probability Threshold",
        "Pipeline Runtime [Sec]",
        "FS Time [Sec]",
        "FS Time [Hr]",
        "Classifier Model",
        "Avg Classifier Time [Sec]",
        "Std_dev Classifier Time [Sec]",
        "Average Accuracy",
        "Std_dev Accuracy",
        "Average AUC",
        "Std_dev AUC",
        "Average Precision",
        "Std_dev Precision",
        "Average Recall",
        "Std_dev Recall",
        "Average F1",
        "Std_dev F1",

        "Max Accuracy (All Folds)",
        "Max AUC (All Folds)",
        "Max Precision (All Folds)",
        "Max Recall (All Folds)",
        "Max F1 (All Folds)",
        "Acc list",
        "AUC list",
        "Prec list",
        "Rec list",
        "F1 list",
        "Best Indices (frequency)",
        "Best Indices (average-rank)",
        "Best Indices (mean-FI)",
    ]


    # If the file exists, read it; otherwise, create a new DataFrame with these columns.
    if os.path.exists(pathresults):
        df = pd.read_excel(pathresults)
        # Optionally, reindex to enforce our desired order:
        # df = df.reindex(columns=columns)
        columns = df.columns.tolist()
    else:
        df = pd.DataFrame(columns=columns)

    return df, columns


def analyze_labels(label_arr):
    # Assume label_arr might have negative values
    label_arr = np.array(label_arr)  # Make sure it's an array if it isn't already
    
    # Use Counter to handle any integer values including negative
    label_counts = Counter(label_arr)
    majority_class = max(label_counts, key=label_counts.get)
    majority_class_count = label_counts[majority_class]
    total_samples = len(label_arr)
    majority_class_percentage = (majority_class_count / total_samples) * 100
    unique_labels = np.unique(label_arr)
    num_unique_labels = len(unique_labels)
    class_fraction = 1 / num_unique_labels if num_unique_labels > 0 else 0
    #print(f"Majority class: {majority_class} with {majority_class_count} occurrences out of {total_samples} samples ({majority_class_percentage:.2f}%).")
    #print(f"Total unique labels: {num_unique_labels}. Class fraction: {class_fraction:.4f}")

    return {
        "majority_class": majority_class,
        "majority_class_count": majority_class_count,
        "majority_class_percentage": majority_class_percentage,
        "total_unique_labels": num_unique_labels,
        "class_fraction": class_fraction
    }




def calculate_bic_performance(df):
    """
    Calculate the Best In Class (BIC) performance for both algorithm and dataset levels.

    This function performs the following steps:
    1. Calculate the maximum 'Average Accuracy' and 'Average AUC' for each combination of 'Repository', 'Dataset', 'Algorithm', 'N', and 'P'.
    2. Add columns to indicate whether 'Average Accuracy' or 'Average AUC' is equal to the maximum values.
    3. Add columns to store unique 'K' and 'Hyperparamter' values for maximum 'Algorithm_Max_Acc' and 'Algorithm_Max_AUC_ovr'.
    4. Fill NaN values with an empty string.
    5. Perform dataset-level aggregation to calculate the maximum 'Average Accuracy' and 'Average AUC'.
    6. Add columns to store unique 'K' and 'Algorithm' values for maximum 'Dataset_Max_Acc' and 'Dataset_Max_AUC_ovr'.
    7. Calculate the minimum 'Std_dev_Accuracy' and 'Std_dev_auc_ovr' for each combination of 'Repository', 'Dataset', 'N', and 'P'.
    8. Flag rows with the minimum 'Std_dev_Accuracy' and 'Std_dev_auc_ovr'.
    9. Add columns for 'FS Acc ± Std' and 'FS AUC ± Std' only for flagged rows.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the necessary columns.

    Returns:
    pd.DataFrame: The updated DataFrame with the new columns added.
    """

    # Algorithm Best In Class (BIC) Performance
    df['Data_Algorithm_5CV_Max_Acc'] = df.groupby(['Repository', 'Dataset', 'Algorithm', 'N', 'P'])['Average Accuracy'].transform('max')
    df['Data_Algorithm_5CV_Max_AUC'] = df.groupby(['Repository', 'Dataset', 'Algorithm', 'N', 'P'])['Average AUC'].transform('max')

    df['Data_Algorithm_5CV_Max_Acc[Y/N]'] = np.where(df['Average Accuracy'] == df['Data_Algorithm_5CV_Max_Acc'], 'Y', 'N')
    df['Data_Algorithm_5CV_Max_AUC[Y/N]'] = np.where(df['Average AUC'] == df['Data_Algorithm_5CV_Max_AUC'], 'Y', 'N')

    # Create temporary DataFrames for the lists of 'K' and 'Hyperparamter' for Accuracy
    temp_ks_acc = df.groupby(['Repository', 'Dataset', 'Algorithm', 'N', 'P']).apply(lambda x: pd.Series({'Data_Algorithm_Best_Ks_Acc': list(x['K'][x['Average Accuracy'] == x['Data_Algorithm_5CV_Max_Acc']].unique())})).reset_index()
    temp_hyperparamter_acc = df.groupby(['Repository', 'Dataset', 'Algorithm', 'N', 'P']).apply(lambda x: pd.Series({'Data_Algorithm_Best_Hyperparamter_Acc': list(x['Hyperparamter'][x['Average Accuracy'] == x['Data_Algorithm_5CV_Max_Acc']].unique())})).reset_index()

    # Create temporary DataFrames for the lists of 'K' and 'Hyperparamter' for AUC
    temp_ks_auc = df.groupby(['Repository', 'Dataset', 'Algorithm', 'N', 'P']).apply(lambda x: pd.Series({'Data_Algorithm_Best_Ks_AUC': list(x['K'][x['Average AUC'] == x['Data_Algorithm_5CV_Max_AUC']].unique())})).reset_index()
    temp_hyperparamter_auc = df.groupby(['Repository', 'Dataset', 'Algorithm', 'N', 'P']).apply(lambda x: pd.Series({'Data_Algorithm_Best_Hyperparamter_AUC': list(x['Hyperparamter'][x['Average AUC'] == x['Data_Algorithm_5CV_Max_AUC']].unique())})).reset_index()

    # Merge the temporary DataFrames back to the original DataFrame
    df = df.merge(temp_ks_acc, on=['Repository', 'Dataset', 'Algorithm', 'N', 'P'], how='left')
    df = df.merge(temp_hyperparamter_acc, on=['Repository', 'Dataset', 'Algorithm', 'N', 'P'], how='left')
    df = df.merge(temp_ks_auc, on=['Repository', 'Dataset', 'Algorithm', 'N', 'P'], how='left')
    df = df.merge(temp_hyperparamter_auc, on=['Repository', 'Dataset', 'Algorithm', 'N', 'P'], how='left')

    # Dataset-level (BIC) aggregation
    df['Data_5CV_Max_Acc'] = df.groupby(['Repository', 'Dataset', 'N', 'P'])['Average Accuracy'].transform('max')
    df['Data_5CV_Max_AUC'] = df.groupby(['Repository', 'Dataset', 'N', 'P'])['Average AUC'].transform('max')

    df['Data_5CV_Max_Acc[Y/N]'] = np.where(df['Average Accuracy'] == df['Data_5CV_Max_Acc'], 'Y', 'N')
    df['Data_5CV_Max_AUC[Y/N]'] = np.where(df['Average AUC'] == df['Data_5CV_Max_AUC'], 'Y', 'N')

    df['Data_Best_Ks_Acc'] = df.groupby(['Repository', 'Dataset', 'N', 'P']).apply(lambda x: list(x['K'][x['Average Accuracy'] == x['Data_5CV_Max_Acc']].unique())).reset_index(level=[0, 1, 2, 3], drop=True)
    df['Data_Best_Algorithm_Acc'] = df.groupby(['Repository', 'Dataset', 'N', 'P']).apply(lambda x: list(x['Algorithm'][x['Average Accuracy'] == x['Data_5CV_Max_Acc']].unique())).reset_index(level=[0, 1, 2, 3], drop=True)

    df['Data_Best_Ks_AUC'] = df.groupby(['Repository', 'Dataset', 'N', 'P']).apply(lambda x: list(x['K'][x['Average AUC'] == x['Data_5CV_Max_AUC']].unique())).reset_index(level=[0, 1, 2, 3], drop=True)
    df['Data_Best_Algorithm_AUC'] = df.groupby(['Repository', 'Dataset', 'N', 'P']).apply(lambda x: list(x['Algorithm'][x['Average AUC'] == x['Data_5CV_Max_AUC']].unique())).reset_index(level=[0, 1, 2, 3], drop=True)

    df['Data_Best_Ks_Acc'] = df['Data_Best_Ks_Acc'].fillna("")
    df['Data_Best_Algorithm_Acc'] = df['Data_Best_Algorithm_Acc'].fillna("")
    df['Data_Best_Ks_AUC'] = df['Data_Best_Ks_AUC'].fillna("")
    df['Data_Best_Algorithm_AUC'] = df['Data_Best_Algorithm_AUC'].fillna("")

    # Calculate the minimum 'Std_dev_Accuracy' for rows with maximum 'Average Accuracy'
    df_max_acc = df[df['Data_5CV_Max_Acc[Y/N]'] == 'Y']
    df_min_std_acc = df_max_acc.groupby(['Repository', 'Dataset', 'N', 'P'])['Std_dev_Accuracy'].transform('min')
    df['Dataset_Min_Std_Acc'] = df.groupby(['Repository', 'Dataset', 'N', 'P'])['Std_dev_Accuracy'].transform(lambda x: x[df['Data_5CV_Max_Acc[Y/N]'] == 'Y'].min())

    # Calculate the minimum 'Std_dev_auc_ovr' for rows with maximum 'Average AUC'
    df_max_auc = df[df['Data_5CV_Max_AUC[Y/N]'] == 'Y']
    df_min_std_auc = df_max_auc.groupby(['Repository', 'Dataset', 'N', 'P'])['Std_dev_auc_ovr'].transform('min')
    df['Dataset_Min_Std_AUC'] = df.groupby(['Repository', 'Dataset', 'N', 'P'])['Std_dev_auc_ovr'].transform(lambda x: x[df['Data_5CV_Max_AUC[Y/N]'] == 'Y'].min())

    # Flag rows with the minimum 'Std_dev_Accuracy' and 'Std_dev_auc_ovr' within the rows with maximum 'Average Accuracy' and 'Average AUC'
    df['Dataset_BIC_Acc_Hyperparamter_K'] = np.where((df['Average Accuracy'] == df['Data_5CV_Max_Acc']) & (df['Std_dev_Accuracy'] == df['Dataset_Min_Std_Acc']), 'Y', '')
    df['Dataset_BIC_AUC_Hyperparamter_K'] = np.where((df['Average AUC'] == df['Data_5CV_Max_AUC']) & (df['Std_dev_auc_ovr'] == df['Dataset_Min_Std_AUC']), 'Y', '')

    # Add columns for 'FS Acc ± Std' and 'FS AUC ± Std' only for flagged rows
    df['FS Acc ± Std'] = np.where(df['Dataset_BIC_Acc_Hyperparamter_K'] == 'Y', df.apply(lambda row: f"{row['Average Accuracy']} ± {row['Std_dev_Accuracy']}", axis=1), "")
    df['FS AUC ± Std'] = np.where(df['Dataset_BIC_AUC_Hyperparamter_K'] == 'Y', df.apply(lambda row: f"{row['Average AUC']} ± {row['Std_dev_auc_ovr']}", axis=1), "")
    
    return df
    
def filter_dataframe(df, data_name, algo_name, rows, columns_len, hyperparamter_comment, perform_cv, initial_splits, train_size):
    """
    Filter the DataFrame based on the specified conditions.

    Args:
    - df (pd.DataFrame): The DataFrame to filter.
    - data_name (str): The name of the dataset.
    - algo_name (str): The name of the algorithm.
    - rows (int): Number of rows in the dataset.
    - columns_len (int): Number of columns (features) in the dataset.
    - hyperparamter_comment (str): The hyperparameter comment.
    - perform_cv (str): Indicates whether cross-validation or split is performed.
    - initial_splits (int): The initial number of splits for cross-validation.
    - train_size (float): The proportion of the dataset to include in the train split.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    if perform_cv == 'Y':
        df_filtered = df[(df['Dataset'] == data_name) &
                        (df['Algorithm'] == algo_name) &
                        (df['N'] == rows) &
                        (df['P'] == columns_len) &
                        (df['Hyperparamter'] == hyperparamter_comment) &
                        (df['Perform_CV[Y/N]'] == perform_cv) &
                        (df['#Folds'] == initial_splits)]
    else:
        df_filtered = df[(df['Dataset'] == data_name) &
                        (df['Algorithm'] == algo_name) &
                        (df['N'] == rows) &
                        (df['P'] == columns_len) &
                        (df['Hyperparamter'] == hyperparamter_comment) &
                        (df['Perform_CV[Y/N]'] == perform_cv) &
                        (df['Train_size%'] == train_size)]
    
    return df_filtered

def generate_run_key(data_name, algo_name, N, P, k, hyper_comment,fixed_threshold):
    """Generate a unique key for a given experiment run."""  
    return f"{data_name}_{algo_name}_{N}_{P}_{k}_{hyper_comment}_{fixed_threshold}"

def check_k_limitation_and_run(
    algo_name,
    hyperparams,
    df_FS,
    data_name,
    rows,
    columns_len,
    fixed_threshold,
    perform_cv,
    initial_splits,
    train_size,
    evaluation_models,
    no_hyperparameters,
    K_List
):
    """
    Check which K values (number of selected features) are valid for the dataset 
    and whether the experiments for those K values and all evaluation models have 
    already been run and recorded in the results DataFrame (df_FS).

    It also checks if a high-dimensional (HD, i.e., No FS) run exists and if all 
    evaluation models were used for that HD run.

    Args:
        algo_name (str): Name of the feature selection algorithm.
        hyperparams (pd.DataFrame): Hyperparameters DataFrame with a 'comment' column.
        df_FS (pd.DataFrame): DataFrame of existing experiment results.
        data_name (str): Dataset name.
        rows (int): Number of rows in the dataset.
        columns_len (int): Number of features in the dataset.
        fixed_threshold (float): Probability threshold used during evaluation.
        perform_cv (str): 'Y' if cross-validation is used, 'N' otherwise.
        initial_splits (int): Number of folds for cross-validation.
        train_size (int or float): Training size percentage if not using CV.
        evaluation_models (list): List of evaluation models that must appear in 'Evaluation Model'.
        no_hyperparameters (list): List of algorithms that do not use hyperparameters.
        K_List (list): List of K values (number of features) to check.

    Returns:
        filtered_k_fs (list): List of K values for which the run has NOT been completed
                              for all evaluation models.
        filtered_k_hd (str): 'Y' if HD (No FS) run exists with all evaluation models,
                            'N' otherwise.
    """

    # Filter K values to those <= number of features
    if columns_len < max(K_List):
        filtered_k = [k for k in K_List if k <= columns_len]
    else:
        filtered_k = K_List.copy()

    # Determine hyperparameter comment string
    hyper_comment = (
        'No_hyperparamters'
        if algo_name in no_hyperparameters
        else hyperparams['comment'].values[0]
    )

    # Build base filter conditions common to all queries
    extra_cond = {'#Folds': initial_splits} if perform_cv.upper() == 'Y' else {'Train_size%': train_size}
    base_conditions = build_filter_conditions(
        data_name, algo_name, rows, columns_len, hyper_comment, perform_cv, fixed_threshold, extra_cond
    )

    # Apply base filtering on the dataframe for efficiency
    df_filtered = df_FS.copy()
    for col, val in base_conditions.items():
        df_filtered = df_filtered[df_filtered[col] == val]

    # Check if HD run exists (No FS) and verify all eval models are run for HD
    filtered_k_hd = 'Y' if 'No FS' in df_filtered['FS/HD'].values else 'N'
    hd_eval_models_run = True
    missing_hd_models = set()
    if filtered_k_hd == 'Y':
        eval_models_hd = set(df_filtered[df_filtered['FS/HD'] == 'No FS']["Classifier Model"].unique())
        missing_hd_models = set(evaluation_models) - eval_models_hd
        if missing_hd_models:
            hd_eval_models_run = False

    # Check for each K if all evaluation models exist
    filtered_k_fs = []
    missing_eval_models_per_k = {}

    for k in filtered_k:
        # Add K to the base filter conditions
        cond_k = base_conditions.copy()
        cond_k['K'] = k

        # Filter dataframe for this K and other conditions
        df_k_filtered = df_FS.copy()
        for col, val in cond_k.items():
            df_k_filtered = df_k_filtered[df_k_filtered[col] == val]

        # Check which evaluation models are missing for this K
        missing_models_for_k = [
            eval_model
            for eval_model in evaluation_models
            if eval_model not in df_k_filtered['Classifier Model'].values
        ]

        if missing_models_for_k:
            filtered_k_fs.append(k)
            missing_eval_models_per_k[k] = missing_models_for_k

    # Print summary information to help user understand what's missing
    if filtered_k_hd == 'Y' and hd_eval_models_run:
        print("You have run all HD (No FS) experiments with all evaluation models.")
    else:
        print("You have NOT run all HD (No FS) experiments or some evaluation models are missing:")
        if filtered_k_hd == 'N':
            print("- No HD run recorded.")
        if not hd_eval_models_run:
            print(f"- Missing HD evaluation models: {missing_hd_models}")

    if not filtered_k_fs:
        print("You have run all FS experiments for all K values and all evaluation models.")
    else:
        print("Missing FS runs detected for the following K values and evaluation models:")
        for k, missing_models in missing_eval_models_per_k.items():
            print(f" - K={k} missing evaluation models: {missing_models}")

    return filtered_k_fs, filtered_k_hd









def build_filter_conditions(
    data_name, algo_name, N, P, hyper_comment, perform_cv, fixed_threshold, extra_cond
):
    conditions = {
        "Dataset": data_name,
        "Algorithm": algo_name,
        "N": N,
        "P": P,
        "Hyperparamter": hyper_comment,
        "Perform_CV[Y/N]": perform_cv,
        "Probability Threshold": fixed_threshold,
    }
    conditions.update(extra_cond)
    return conditions





def ensure_columns(df, final_columns):
    """
    Guarantee all final_columns exist; create missing as NaN.
    Keeps other columns too (you can reorder/select later).
    """
    df = df.copy()
    for c in final_columns:
        if c not in df.columns:
            df[c] = np.nan
    # Optional: reorder to final_columns first, then the rest
    ordered = final_columns + [c for c in df.columns if c not in final_columns]
    return df.loc[:, ordered]

def process_fs_algorithm(label_arr,specific_experiment,fixed_threshold, data_name, algo_name, filtered_k, dataset_path, hyperparamters, repository, columns_len,  path_output_excel_FS, alg, columns,FS_HD,debug):

    """
    Process the feature selection algorithm and save the results.

    Args:
    - label_arr (np.ndarray): The label array.
    - data_name (str): The name of the dataset.
    - algo_name (str): The name of the algorithm.
    - filtered_k (list): List of filtered K values.
    - dataset_path (str): The path to the dataset.
    - hyperparamters (pd.DataFrame): DataFrame containing hyperparameters.
    - repository (str): The repository name.
    - columns_len (int): Number of columns (features) in the dataset.
    - path_output_excel_FS (str): The path to the output Excel directory.
    - alg (object): The algorithm object.
    - columns (list): List of columns for the DataFrame.

    Returns:
    - None
    """
    results_dataset = analyze_labels(label_arr)
    print(f"\033[1mStart FS for Dataset:\033[0m {data_name}\n"
          f"\033[1mFor FS_Algorithm when K=P :\033[0m {algo_name}\n")

    # Create the context dictionary
    context = {
        'FS_HD': FS_HD,
        'dataset_path': dataset_path,
        'hyperparamters': hyperparamters,
        'algo_name': algo_name,
        'repository': repository,
        'data_name': data_name,
        'columns_len': columns_len,
    }


    # Call run_parallel_processing with the context and filtered_k
    results = run_parallel_processing(context,fixed_threshold, filtered_k,debug)
    
    if results:  # Only concatenate if results is not empty.
        
        df = pd.concat(results, ignore_index=True)
        df['Majority_class_percentage'] = round(results_dataset['majority_class_percentage'] / 100, 2)
        df['#Classes'] = results_dataset['total_unique_labels']
        df['Class_Fraction'] = round(results_dataset['class_fraction'], 2)
        df['FS/HD'] = FS_HD
        df['Perform_CV[Y/N]'] = perform_cv

        if perform_cv == 'Y':
            df['Train_size%'] = 1- (1 / (df['#Successfuls_folds'].astype(float)))
        else:
            df['Train_size%'] = train_size



        # Dataset Hardness Label
        #df = dataset_hardness_score(df, HighT, LowT)
        df['Dataset'] = df['Dataset'].str.replace(r'_\d+_samples$', '', regex=True)




        df = df[columns]

        print(f"Finish {data_name} with {FS_HD}")



        
        # Save output to Excel...
        output_file = os.path.join(path_output_excel_FS, f"{specific_experiment}.xlsx")

        if os.path.exists(output_file):
            old_output = pd.read_excel(output_file)
            output_All = pd.concat([old_output, df])
            output_All  = output_All .sort_values([ 'Dataset', 'Algorithm','K',]).drop_duplicates(['Data_Algo_rows_columns_k_Hyperparamter_Threshold_Evaluator_Key','Perform_CV[Y/N]','#Folds','FS/HD'])
        else:
            output_All  = df.copy()

        output_All  = output_All .reset_index(drop=True)
        output_All .to_excel(path_output_excel_FS + specific_experiment + '.xlsx', index=False)
        

        #if FS_HD =='FS' :
        #    # Call the calculate_bic_performance function
        #    df_summary = calculate_bic_by_dataset_fs(output_All)
        #    #df = calculate_bic_performance(df)


def calculate_bic_by_dataset_fs(df):
    """
    Computes the best-in-class summary per group (Dataset and FS/HD) for several metrics,
    then restructures the summary into a long-format table that includes, for each dataset
    and metric, the FS and No FS performance in a formatted string, extracts the Algorithm
    and Hyperparameter from the corresponding key column, and includes the raw average 
    values and standard deviations.
    
    Only datasets that have both FS and No FS rows (in column "FS/HD") are included.
    
    When more than one algorithm ties for best performance, their keys are concatenated
    (separated by " ; "), and the extraction function will then concatenate the extracted 
    algorithms and hyperparameters similarly.
    
    Returns:
        pd.DataFrame: A new summary DataFrame with one row per dataset per metric, containing:
                      - Dataset
                      - Metric (short name)
                      - No FS: formatted string "Average ± Std (Category)"
                      - FS: formatted string "Average ± Std (Category)"
                      - Dataset Category (determined by comparing FS and No FS performance)
                      - No FS Algorithm, No FS Hyperparameter, FS Algorithm, FS Hyperparameter
                      - No FS Average, FS Average (raw average values formatted as percentages)
                      - No FS Std, FS Std (raw standard deviations formatted as percentages)
    """




    # Performance labeling using thresholds.
    HighT = 0.9
    LowT = 0.7
    def performance_label(metric):
        if metric > HighT:
            return "High"
        elif metric < LowT:
            return "Low"
        else:
            return "Fair"

    # Mapping from each average metric to its corresponding std column.
    metrics = {
        'Average Accuracy': 'Std_dev Accuracy',
        'Average AUC': 'Std_dev AUC',
        'Average Precision': 'Std_dev Precision',
        'Average Recall': 'Std_dev Recall',
        'Average F1': 'Std_dev F1'
    }

    # Group by 'Dataset' and 'FS/HD' and compute best-in-class summaries.
    group_cols = ['Dataset', 'FS/HD']
    summary_list = []
    
    for name, group in df.groupby(group_cols):
        summary = dict(zip(group_cols, name))
        for avg_metric, std_col in metrics.items():
            # Find the maximum value for the metric in the group.
            max_val = group[avg_metric].max()
            # Filter rows with the maximum value.
            best_rows = group[group[avg_metric] == max_val]
            # Among these, select the row(s) with the minimum standard deviation.
            min_std = best_rows[std_col].min()
            best_rows_min_std = best_rows[best_rows[std_col] == min_std]
            # Concatenate all keys if more than one row ties.
            keys = best_rows_min_std['Data_Algo_rows_columns_k_Hyperparamter_Threshold_Evaluator_Key'].tolist()
            combined_key = " ; ".join(keys)
            
            summary[f'BIC_{avg_metric}'] = custom_round(max_val)
            summary[f'{avg_metric} Performance'] = performance_label(max_val)
            summary[f'BIC_{std_col}'] = custom_round(min_std)
            summary[f'BIC_{avg_metric}_Key'] = combined_key
        summary_list.append(summary)
    
    df_summary_dataset = pd.DataFrame(summary_list)
    
    # Rename group label "HD" to "No FS" to ensure consistency.
    df_summary_dataset['FS/HD'] = df_summary_dataset['FS/HD'].replace({'HD': 'No FS'})
    
    # -------------------------------
    # 1. Remove datasets that do not have both FS and No FS rows.
    datasets_to_keep = []
    for ds in df_summary_dataset['Dataset'].unique():
        df_ds = df_summary_dataset[df_summary_dataset['Dataset'] == ds]
        if "FS" in df_ds['FS/HD'].values and "No FS" in df_ds['FS/HD'].values:
            datasets_to_keep.append(ds)
    df_summary_dataset = df_summary_dataset[df_summary_dataset['Dataset'].isin(datasets_to_keep)]
    
    # -------------------------------
    # 2. Helper function to extract Algorithm and Hyperparameter from a key.
    def extract_algo_and_hyperparam(key):
        # If multiple keys are concatenated with " ; ", process each separately.
        if " ; " in key:
            keys = key.split(" ; ")
            algorithms = []
            hyperparameters = []
            for k in keys:
                parts = k.split('_')
                if len(parts) >= 6:
                    algorithms.append(parts[1])
                    hyperparameters.append("_".join(parts[5:]))
            return " ; ".join(algorithms), " ; ".join(hyperparameters)
        else:
            parts = key.split('_')
            if len(parts) >= 6:
                algorithm = parts[1]
                hyperparameter = "_".join(parts[5:])
            else:
                algorithm = None
                hyperparameter = None
            return algorithm, hyperparameter

    # Function to determine the dataset category based on FS and No FS performance.
    order = {"Low": 1, "Fair": 2, "High": 3}
    def determine_category(nofs_label, fs_label):
        # If FS performance is lower than No FS, mark as Fragile.
        if order[fs_label] < order[nofs_label]:
            return "Fragile"
        else:
            # Otherwise, use FS performance to define the category.
            if fs_label == "High":
                return "Easy"
            elif fs_label == "Fair":
                return "Medium"
            elif fs_label == "Low":
                return "Hard"
    
    # -------------------------------
    # Restructure the summary table into a long-format table.
    new_rows = []
    datasets = df_summary_dataset['Dataset'].unique()
    for ds in datasets:
        df_ds = df_summary_dataset[df_summary_dataset['Dataset'] == ds]
        # Only process datasets that have both FS and No FS.
        if not ("FS" in df_ds['FS/HD'].values and "No FS" in df_ds['FS/HD'].values):
            continue
        row_nofs = df_ds[df_ds['FS/HD'] == "No FS"].iloc[0]
        row_fs = df_ds[df_ds['FS/HD'] == "FS"].iloc[0]
        
        for avg_metric, std_suffix in metrics.items():
            # Retrieve best values and standard deviations.
            value_nofs = row_nofs["BIC_" + avg_metric]
            std_nofs = row_nofs["BIC_" + std_suffix]
            perf_nofs = row_nofs[avg_metric + " Performance"]

            value_fs = row_fs["BIC_" + avg_metric]
            std_fs = row_fs["BIC_" + std_suffix]
            perf_fs = row_fs[avg_metric + " Performance"]

            # Format the performance string (assuming values are in decimal form).
            nofs_str = f"{value_nofs*100:.1f}% ± {std_nofs*100:.2f}% ({perf_nofs})"
            fs_str = f"{value_fs*100:.1f}% ± {std_fs*100:.2f}% ({perf_fs})"

            # Determine dataset category for this metric.
            ds_category = determine_category(perf_nofs, perf_fs)

            # Short metric name (e.g., "Accuracy" from "Average Accuracy")
            metric_short = avg_metric.split()[1]
            
            # Extract Algorithm and Hyperparameter from the key.
            key_col = "BIC_" + avg_metric + "_Key"
            algo_nofs, hyper_noths = extract_algo_and_hyperparam(row_nofs[key_col])
            algo_fs, hyper_fs = extract_algo_and_hyperparam(row_fs[key_col])
            
            # Add raw average values and standard deviations as percentages.
            nofs_avg = f"{value_nofs*100:.1f}%"
            fs_avg = f"{value_fs*100:.1f}%"
            nofs_std = f"{std_nofs*100:.2f}%"
            fs_std = f"{std_fs*100:.2f}%"
            
            new_rows.append({
                "Dataset": ds,
                "Metric": metric_short,
                "No FS": nofs_str,
                "FS": fs_str,
                "Dataset Category": ds_category,
                "No FS Algorithm": algo_nofs,
                "No FS Hyperparameter": hyper_noths if (hyper_noths := hyper_noths) else hyper_noths,  # variable consistency
                "FS Algorithm": algo_fs,
                "FS Hyperparameter": hyper_fs,
                "No FS Average": nofs_avg,
                "FS Average": fs_avg,
                "No FS Std": nofs_std,
                "FS Std": fs_std
            })
    
    df_new_structure = pd.DataFrame(new_rows)
    
    # Optionally, output the new structure to Excel (ensure path_output_excel_FS is defined).
    output_file = os.path.join(path_output_excel_FS, "Results_Summary.xlsx")
    df_new_structure.to_excel(output_file, index=False)
    
    return df_new_structure


def process_input(k, fixed_threshold, context, task_index):
    """
    Process input data for feature selection or high-dimensional analysis.
    """
    import scipy.io
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    import torch

    FS_HD        = context['FS_HD']
    dataset_path = context['dataset_path']
    hyperparamters = context['hyperparamters']
    algo_name    = context['algo_name']
    columns_len  = context['columns_len']
    repository   = context['repository']
    data_name    = context['data_name']

    # Whether this task SHOULD use GPU (decided in run_parallel_processing)
    use_gpu = context.get("use_gpu", False)

    def tf_device_from_torch(device):
        if device.type == 'cuda':
            return '/device:GPU:0'
        if device.type == 'mps':
            gpus = tf.config.list_logical_devices('GPU')
            return gpus[0].name if gpus else '/device:CPU:0'
        return '/device:CPU:0'

    # Pick the torch device based on use_gpu flag
    if use_gpu and torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device('mps')   # Apple Silicon
        else:
            device = torch.device('cuda')  # CUDA
    else:
        device = torch.device('cpu')

    print(f"[Task {task_index}] {algo_name} → {device.type.upper()} (use_gpu={use_gpu})")

    tf_device = tf_device_from_torch(device)

    repository = context['repository']
    data_name = context['data_name']
    columns_len = context['columns_len']

    # Load and preprocess data once.
    Data = scipy.io.loadmat(dataset_path)
    data_arr = np.abs(Data['X'])
    label_arr = Data['Y'][:, 0] if min(Data['Y'][:, 0]) == 0 else Data['Y'][:, 0] - 1
    # Scale data once.
    scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_arr)
    
    
    P = columns_len
    df = pd.DataFrame()
    
    # For neural network methods, pass the device parameter along.
    if algo_name in no_hyperparameters:
        row = hyperparamters  # Dummy value for algorithms with no hyperparameters.
        if perform_cv.upper() == 'Y':
            result_df  = run_cross_validation(fixed_threshold,repository,data_name,P,data_arr, label_arr,  row, k, algo_name,hyperparamters,FS_HD,  device=device)
                        
        else:
            random_state = seed
            fold_train_data, fold_test_data, fold_train_labels, fold_test_labels = train_test_split(
                scaled_data, label_arr, test_size=(1-train_size), stratify=label_arr, random_state=random_state
            )
            fold_train_data, fold_test_data, fold_train_labels, fold_test_labels = fill_missing_labels(
                fold_train_data, fold_test_data, fold_train_labels, fold_test_labels
            )
            result_df, best_accuracy, best_AUC_ovo, best_AUC_ovr = run_train_test_split(
                specific_experiment, FS_HD, fold_train_data, fold_train_labels, fold_test_data,
                fold_test_labels, data_name, row, k, P, repository, algo_name, train_size
            )
        df = pd.concat([result_df, df], ignore_index=True)
    else:
        # For algorithms with hyperparameters, iterate over each hyperparameter setting.
        for index, row in hyperparamters.iterrows():
            # For neural network methods, pass the 'device' parameter to the feature selection function.
            if algo_name in algo_name_NN :
                # The feature selection function signature is adjusted to accept a device.
                if perform_cv.upper() == 'Y':
                    result_df  = run_cross_validation(fixed_threshold,repository,data_name,P,data_arr, label_arr,  row, k, algo_name,hyperparamters,FS_HD,  device=device)
                else:
                    random_state = seed
                    fold_train_data, fold_test_data, fold_train_labels, fold_test_labels = train_test_split(
                        scaled_data, label_arr, test_size=(1-train_size), stratify=label_arr, random_state=random_state
                    )
                    fold_train_data, fold_test_data, fold_train_labels, fold_test_labels = fill_missing_labels(
                        fold_train_data, fold_test_data, fold_train_labels, fold_test_labels
                    )
                    result_df= run_train_test_split(
                        specific_experiment, FS_HD, fold_train_data, fold_train_labels, fold_test_data,
                        fold_test_labels, data_name, row, k, P, repository, algo_name, train_size, device=device
                    )
            else:
                # For non–neural network methods, device is not used.
                if perform_cv.upper() == 'Y':
                    result_df  = run_cross_validation(fixed_threshold,repository,data_name,P,data_arr, label_arr,  row, k, algo_name,hyperparamters,FS_HD,  device=device)
                else:
                    random_state = seed
                    fold_train_data, fold_test_data, fold_train_labels, fold_test_labels = train_test_split(
                        scaled_data, label_arr, test_size=(1-train_size), stratify=label_arr, random_state=random_state
                    )
                    fold_train_data, fold_test_data, fold_train_labels, fold_test_labels = fill_missing_labels(
                        fold_train_data, fold_test_data, fold_train_labels, fold_test_labels
                    )
                    result_df= run_train_test_split(
                        specific_experiment, FS_HD, fold_train_data, fold_train_labels, fold_test_data,
                        fold_test_labels, data_name, row, k, P, repository, algo_name, train_size
                    )
            df = pd.concat([result_df, df], ignore_index=True)
    return df






def convert_labels_to_running_numbers(labels):
    """
    Convert labels to running numbers from the lowest to the highest value.

    Args:
        labels (numpy array): Array of labels.

    Returns:
        numpy array: Labels converted to running numbers.
    """
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    converted_labels = np.vectorize(label_mapping.get)(labels)
    return converted_labels



def run_parallel_processing(context, fixed_threshold, k_list, debug=False):
    """
    Run processing for each K in parallel on CPU‐only algos,
    or serially on GPU‐eligible algos.
    """
    import multiprocessing as mp
    import torch
    from joblib import Parallel, delayed

    ks        = k_list if context['FS_HD'] == 'FS' else [context['columns_len']]
    algo_name = context['algo_name']

    # Debug mode: run serially in-process with printouts. - update in config.py
    if debug:
        results = []
        for idx, k in enumerate(ks):
            print(f"[DEBUG] Running K={k} idx={idx} in-process")
            res = process_input(k, fixed_threshold, context, idx)
            results.append(res)
        return results


    # Decide once: is this algo allowed to use GPU?
    gpu_available = torch.cuda.is_available()
    use_gpu       = (algo_name in GPU_ALGOS) and gpu_available

    # propagate this decision into the context
    context = {**context, "use_gpu": use_gpu}

    # If this algo can use the GPU, run serially so you don't
    # spawn N CPU processes all fighting over the same GPU:
    if use_gpu:
        results = []
        for idx, k in enumerate(ks):
            results.append(process_input(k, fixed_threshold, context, idx))
        return results

    # Otherwise, dispatch to a CPU pool:
    n_jobs = mp.cpu_count() - 1 or 1
    return Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_input)(k, fixed_threshold, context, idx)
        for idx, k in enumerate(ks)
    )




def construct_W(X, **kwargs):
    """
    Constructs an affinity matrix based on input features and provided options.

    Args:
        X (numpy.ndarray): Input data.
        kwargs (dict): Additional parameters such as metric, neighbor mode, weight mode.

    Returns:
        scipy.sparse.csc_matrix: Constructed affinity matrix.
    """
    # default metric is 'cosine'
    from sklearn.metrics.pairwise import pairwise_distances
    import scipy.sparse


    if 'metric' not in kwargs.keys():
        kwargs['metric'] = 'cosine'

    # default neighbor mode is 'knn' and default neighbor size is 5
    if 'neighbor_mode' not in kwargs.keys():
        kwargs['neighbor_mode'] = 'knn'
    if kwargs['neighbor_mode'] == 'knn' and 'k' not in kwargs.keys():
        kwargs['k'] = k # 5
    if kwargs['neighbor_mode'] == 'supervised' and 'k' not in kwargs.keys():
        kwargs['k'] = k # 5 
    if kwargs['neighbor_mode'] == 'supervised' and 'y' not in kwargs.keys():
        print('Warning: label is required in the supervised neighborMode!!!')
        exit(0)

    # default weight mode is 'binary', default t in heat kernel mode is 1
    if 'weight_mode' not in kwargs.keys():
        kwargs['weight_mode'] = 'binary'
    if kwargs['weight_mode'] == 'heat_kernel':
        if kwargs['metric'] != 'euclidean':
            kwargs['metric'] = 'euclidean'
        if 't' not in kwargs.keys():
            kwargs['t'] = 1
    elif kwargs['weight_mode'] == 'cosine':
        if kwargs['metric'] != 'cosine':
            kwargs['metric'] = 'cosine'

    # default fisher_score and ReliefF mode are 'false'
    if 'fisher_score' not in kwargs.keys():
        kwargs['fisher_score'] = False
    if 'ReliefF' not in kwargs.keys():
        kwargs['ReliefF'] = False

    n_samples, n_features = np.shape(X)

    # choose 'knn' neighbor mode
    if kwargs['neighbor_mode'] == 'knn':
        k = kwargs['k']
        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                # compute pairwise euclidean distances
                D = pairwise_distances(X)
                D **= 2
                # sort the distance matrix D in ascending order
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                # choose the k-nearest neighbors for each instance
                idx_new = idx[:, 0:k + 1]
                G = np.zeros((n_samples * (k + 1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order='F')
                G[:, 2] = 1
                # build the sparse affinity matrix W
                W = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

            elif kwargs['metric'] == 'cosine':
                # normalize the data first
                X_normalized = np.power(np.sum(X * X, axis=1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :] / max(1e-12, X_normalized[i])
                # compute pairwise cosine distances
                D_cosine = np.dot(X, np.transpose(X))
                # sort the distance matrix D in descending order
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0:k + 1]
                G = np.zeros((n_samples * (k + 1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order='F')
                G[:, 2] = 1
                # build the sparse affinity matrix W
                W = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            t = kwargs['t']
            # compute pairwise euclidean distances
            D = pairwise_distances(X)
            D **= 2
            # sort the distance matrix D in ascending order
            dump = np.sort(D, axis=1)
            idx = np.argsort(D, axis=1)
            idx_new = idx[:, 0:k + 1]
            dump_new = dump[:, 0:k + 1]
            # compute the pairwise heat kernel distances
            dump_heat_kernel = np.exp(-dump_new / (2 * t * t))
            G = np.zeros((n_samples * (k + 1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
            G[:, 1] = np.ravel(idx_new, order='F')
            G[:, 2] = np.ravel(dump_heat_kernel, order='F')
            # build the sparse affinity matrix W
            W = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

        elif kwargs['weight_mode'] == 'cosine':
            # normalize the data first
            X_normalized = np.power(np.sum(X * X, axis=1), 0.5)
            for i in range(n_samples):
                X[i, :] = X[i, :] / max(1e-12, X_normalized[i])
            # compute pairwise cosine distances
            D_cosine = np.dot(X, np.transpose(X))
            # sort the distance matrix D in ascending order
            dump = np.sort(-D_cosine, axis=1)
            idx = np.argsort(-D_cosine, axis=1)
            idx_new = idx[:, 0:k + 1]
            dump_new = -dump[:, 0:k + 1]
            G = np.zeros((n_samples * (k + 1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)
            G[:, 1] = np.ravel(idx_new, order='F')
            G[:, 2] = np.ravel(dump_new, order='F')
            # build the sparse affinity matrix W
            W = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

    # choose supervised neighborMode
    elif kwargs['neighbor_mode'] == 'supervised':
        k = kwargs['k']
        # get true labels and the number of classes
        y = kwargs['y']
        label = np.unique(y)
        n_classes = np.unique(y).size
        # construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
        if kwargs['fisher_score'] is True:
            W = scipy.sparse.lil_matrix((n_samples, n_samples))
            for i in range(n_classes):
                class_idx = (y == label[i])
                class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
                W[class_idx_all] = 1.0 / np.sum(np.sum(class_idx))
            return W

        # construct the weight matrix W in a ReliefF way, NH(x) and NM(x,y) denotes a set of k nearest
        # points to x with the same class as x, a different class (the class y), respectively. W_ij = 1 if i = j;
        # W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y)
        if kwargs['ReliefF'] is True:
            # when xj in NH(xi)
            G = np.zeros((n_samples * (k + 1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                D = pairwise_distances(X[class_idx, :])
                D **= 2
                idx = np.argsort(D, axis=1)
                idx_new = idx[:, 0:k + 1]
                n_smp_class = (class_idx[idx_new[:]]).size
                if len(class_idx) <= k:
                    k = len(class_idx) - 1
                G[id_now:n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                G[id_now:n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class + id_now, 2] = 1.0 / k
                id_now += n_smp_class
            W1 = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            # when i = j, W_ij = 1
            for i in range(n_samples):
                W1[i, i] = 1
            # when x_j in NM(x_i, y)
            G = np.zeros((n_samples * k * (n_classes - 1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx1 = np.column_stack(np.where(y == label[i]))[:, 0]
                X1 = X[class_idx1, :]
                for j in range(n_classes):
                    if label[j] != label[i]:
                        class_idx2 = np.column_stack(np.where(y == label[j]))[:, 0]
                        X2 = X[class_idx2, :]
                        D = pairwise_distances(X1, X2)
                        idx = np.argsort(D, axis=1)
                        idx_new = idx[:, 0:k]
                        n_smp_class = len(class_idx1) * k
                        G[id_now:n_smp_class + id_now, 0] = np.tile(class_idx1, (k, 1)).reshape(-1)
                        G[id_now:n_smp_class + id_now, 1] = np.ravel(class_idx2[idx_new[:]], order='F')
                        G[id_now:n_smp_class + id_now, 2] = -1.0 / ((n_classes - 1) * k)
                        id_now += n_smp_class
            W2 = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W2) > W2
            W2 = W2 - W2.multiply(bigger) + np.transpose(W2).multiply(bigger)
            W = W1 + W2
            return W

        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                G = np.zeros((n_samples * (k + 1), 3))
                id_now = 0
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise euclidean distances for instances in class i
                    D = pairwise_distances(X[class_idx, :])
                    D **= 2
                    # sort the distance matrix D in ascending order for instances in class i
                    idx = np.argsort(D, axis=1)
                    idx_new = idx[:, 0:k + 1]
                    n_smp_class = len(class_idx) * (k + 1)
                    G[id_now:n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                    G[id_now:n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                    G[id_now:n_smp_class + id_now, 2] = 1
                    id_now += n_smp_class
                # build the sparse affinity matrix W
                W = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

            if kwargs['metric'] == 'cosine':
                # normalize the data first
                X_normalized = np.power(np.sum(X * X, axis=1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :] / max(1e-12, X_normalized[i])
                G = np.zeros((n_samples * (k + 1), 3))
                id_now = 0
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise cosine distances for instances in class i
                    D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                    # sort the distance matrix D in descending order for instances in class i
                    idx = np.argsort(-D_cosine, axis=1)
                    idx_new = idx[:, 0:k + 1]
                    n_smp_class = len(class_idx) * (k + 1)
                    G[id_now:n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                    G[id_now:n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                    G[id_now:n_smp_class + id_now, 2] = 1
                    id_now += n_smp_class
                # build the sparse affinity matrix W
                W = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            G = np.zeros((n_samples * (k + 1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                # compute pairwise cosine distances for instances in class i
                D = pairwise_distances(X[class_idx, :])
                D **= 2
                # sort the distance matrix D in ascending order for instances in class i
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                idx_new = idx[:, 0:k + 1]
                dump_new = dump[:, 0:k + 1]
                t = kwargs['t']
                # compute pairwise heat kernel distances for instances in class i
                dump_heat_kernel = np.exp(-dump_new / (2 * t * t))
                n_smp_class = len(class_idx) * (k + 1)
                G[id_now:n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                G[id_now:n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class + id_now, 2] = np.ravel(dump_heat_kernel, order='F')
                id_now += n_smp_class
            # build the sparse affinity matrix W
            W = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

        elif kwargs['weight_mode'] == 'cosine':
            # normalize the data first
            X_normalized = np.power(np.sum(X * X, axis=1), 0.5)
            for i in range(n_samples):
                X[i, :] = X[i, :] / max(1e-12, X_normalized[i])
            G = np.zeros((n_samples * (k + 1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                # compute pairwise cosine distances for instances in class i
                D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                # sort the distance matrix D in descending order for instances in class i
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0:k + 1]
                dump_new = -dump[:, 0:k + 1]
                n_smp_class = len(class_idx) * (k + 1)
                G[id_now:n_smp_class + id_now, 0] = np.tile(class_idx, (k + 1, 1)).reshape(-1)
                G[id_now:n_smp_class + id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class + id_now, 2] = np.ravel(dump_new, order='F')
                id_now += n_smp_class
            # build the sparse affinity matrix W
            W = scipy.sparse.csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W
        

def calculate_feature_importance_rf(x, y, selected_indices, random_state):
    from sklearn.ensemble import ExtraTreesClassifier

    """
    Calculates feature importances using ExtraTreesClassifier for the selected features.

    Args:
    - x (numpy.ndarray): The feature matrix of selected features.
    - y (numpy.ndarray): The labels.
    - selected_indices (list): Indices of the selected features.
    - random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
    - tuple: (sorted_indices, sorted_importance_rounded, accumulated_importance)
        - sorted_indices (list of int): Sorted indices based on importance.
        - sorted_importance_rounded (list of float): Rounded importance scores.
        - accumulated_importance (float): Sum of rounded importance scores.
    """

    # Ensure x and y are numpy arrays
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    clf = ExtraTreesClassifier(n_estimators=200, max_depth=3, random_state=random_state)
    clf.fit(x, y)
    importances = clf.feature_importances_
    selected_importances = importances[selected_indices]
    normalized_importances = selected_importances / np.sum(importances)
    #normalized_importances = normalized_importances[selected_indices]
    sorted_indices_importance = sorted(zip(selected_indices, normalized_importances), key=lambda x: x[1], reverse=True)
    sorted_indices, sorted_importance = zip(*sorted_indices_importance)
    sorted_importance_rounded = np.array([custom_round(val) for val in sorted_importance])
    accumulated_importance = custom_round(np.sum(sorted_importance_rounded))

    if accumulated_importance > 1:
        accumulated_importance = 1  # Adjust if sum exceeds 1 due to rounding

    return list(sorted_indices), list(sorted_importance_rounded), accumulated_importance



def create_precision_recall_shap_figure(fold_test_labels, fold_test_predicted_probs, fold_test_data_selected, clf, path_output_excel_FS, repository, indices_for_selected,specific_experiment,FS_HD,data_name,k,P):
   
    """
    Create and save the precision-recall figure.

    Args:
    - fold_test_labels (array): True labels for the test set.
    - fold_test_predicted_probs (array): Predicted probabilities for the test set.
    - path_output_excel_FS (str): Path to the output directory.
    - repository (str): Name of the repository.
    """
    

    if specific_experiment == 'MCI':
        path_output_excel_FS = path_output_excel_MCI

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(fold_test_labels, fold_test_predicted_probs)
    
    # Create precision-recall display
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    
    # Plot precision-recall curve
    pr_display.plot()
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # Define the path to save the figure
    precision_recall_folder = os.path.join(path_output_excel_FS, "Precision Recall and Shap Curves")
    if not os.path.exists(precision_recall_folder):
        os.makedirs(precision_recall_folder)
    
    dataset_path = os.path.join(precision_recall_folder, repository, "Original_all samples_80", data_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    figure_path = os.path.join(dataset_path, f"{FS_HD}_K_{k}_P_{P}_precision_recall_curve.png")
    
    # Save the precision-recall figure
    plt.savefig(figure_path)
    plt.close()

    print(f"Precision-Recall curve saved to {figure_path}")

    # Convert indices_for_selected to strings
    column_names = [str(i) for i in indices_for_selected]

    # Convert fold_test_data_selected to a DataFrame with the specified column names
    fold_test_data_selected_df = pd.DataFrame(fold_test_data_selected, columns=column_names)

    # Features and target
    X = fold_test_data_selected_df

    # Create a SHAP explainer and compute SHAP values
    explainer = shap.Explainer(clf, X)
    shap_values = explainer(X)

    # Print shapes for debugging
    print(f"Shape of X: {X.shape}")
    print(f"Shape of shap_values: {shap_values.shape}")

    # Select the appropriate output dimension for multi-output models
    shap_values_single_output = shap_values[..., 0]

    # Detailed waterfall plot for the first prediction
    plt.figure(figsize=(10, 6))  # Adjust the figure size
    shap_values_single_output_0 = shap_values_single_output[0]
    shap_values_single_output_0.feature_names = column_names  # Set feature names to indices only
    shap_values_single_output_0.data = None  # Remove feature values from the plot
    shap.plots.waterfall(shap_values_single_output_0, show=False)
    waterfall_figure_path = os.path.join(dataset_path, f"{FS_HD}_K_{k}_P_{P}_shap_waterfall_plot.png")
    plt.title(f'SHAP Waterfall Plot (K = {k})')
    plt.savefig(waterfall_figure_path, bbox_inches='tight')  # Ensure the figure fits within the saved image
    plt.clf()
    print(f"SHAP waterfall plot saved to {waterfall_figure_path}")







# consensus_selectors  :
# ================================================================
# Helper functions for choosing a single, “consensus” K‑feature set
# after cross‑validation.
# ================================================================

from collections import Counter, defaultdict
from itertools   import chain
import numpy as np


# ------------------------------------------------------------------
# 1. Pure‑frequency voting
# ------------------------------------------------------------------
def consensus_frequency(selected_indices_list, k):
    """
    Select the K most frequently chosen feature indices.

    Logic
    -----
    1. Flatten the per‑fold lists into one long list.
    2. Count how many times each index appears.
    3. Take the K indices with the highest count.
       If < K unique indices ever appeared, pad with the earliest
       indices from the first fold.

    Pros
    ----
    • Very fast – O(total_selected).  
    • Extremely easy to explain (“most stable across folds”).

    Cons
    ----
    • Ignores the within‑fold ordering/importance.  
    • A feature that appears many times at rank K can outrank one that
      appears fewer times at rank 1.
    """
    all_indices  = list(chain.from_iterable(selected_indices_list))
    freq_counter = Counter(all_indices)

    best_indices = [idx for idx, _ in freq_counter.most_common(k)]

    # pad if fewer than K unique indices observed
    if len(best_indices) < k:
        for idx in selected_indices_list[0]:
            if idx not in best_indices:
                best_indices.append(idx)
            if len(best_indices) == k:
                break
    return best_indices


# ------------------------------------------------------------------
# 2. Average‑rank aggregation  (left‑to‑right ⇒ decreasing importance)
# ------------------------------------------------------------------
def consensus_average_rank(selected_indices_list, k):
    """
    Select K indices with the lowest *average rank* across CV folds.

    Assumption
    ----------
    In every per‑fold list, the **left‑most element is rank 1**
    (highest importance), the next is rank 2, and so on.

    Algorithm
    ---------
    1. For each fold, assign ranks based on position (rank 1 = index 0).  
    2. Accumulate the ranks received by each feature over all folds.  
    3. Compute the mean rank per feature.  
    4. Return the K features with the *lowest* average rank
       (i.e., those that most often appear near the top).

    Pros
    ----
    • Captures within‑fold ordering – more nuanced than pure frequency.  
    • Computationally cheap; no extra model evaluations.

    Cons
    ----
    • Requires each per‑fold list to be **ordered** by importance.  
    • If your FS algorithm returns an unordered set, you must impose an
      order first (e.g., by SHAP or permutation importance).
    """
    rank_dict = defaultdict(list)  # {feature_idx: [ranks]}

    # Step 1+2: record ranks
    for fold_list in selected_indices_list:
        for rank, idx in enumerate(fold_list, start=1):  # rank 1 = left‑most
            rank_dict[idx].append(rank)

    # Step 3: mean rank per feature
    avg_rank = {idx: np.mean(ranks) for idx, ranks in rank_dict.items()}

    # Step 4: choose K smallest means  →  most consistently top‑ranked
    best_indices = sorted(avg_rank, key=avg_rank.get)[:k]
    return best_indices


# ------------------------------------------------------------------
# 3. Mean‑importance aggregation  (SHAP or FI)
# ------------------------------------------------------------------
def consensus_mean_importance(importance_dict, k):
    """
    Select K indices with the highest *mean absolute importance*.

    Parameters
    ----------
    importance_dict : dict[int, list[float]]
        Mapping  {feature_idx: [importance_fold1, importance_fold2, ...]}.
        Works for SHAP values, permutation importance, ExtraTrees FI, etc.
    k : int
        Number of features to return.

    Pros
    ----
    • Directly ties selection to the magnitude of a feature’s influence.  
    • A feature with huge importance in one fold can still be selected
      even if it appears rarely.

    Cons
    ----
    • Requires you to collect and store per‑fold importance values.  
    • Slightly more plumbing: you must pass the importance dictionary
      out of the CV loop to this function.
    """
    mean_imp     = {idx: np.mean(vals) for idx, vals in importance_dict.items()}
    best_indices = sorted(mean_imp, key=mean_imp.get, reverse=True)[:k]
    return best_indices



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_eval_model(name, seed):
    """
    Return a fresh classifier for unbiased *evaluation only*.
    IMPORTANT: keep this separate from the FS step to avoid circular validation.
    """
    name = name.upper()

    if name == "LR":
        # Simple, interpretable baseline
        return LogisticRegression(max_iter=200, random_state=seed)

    if name == "SVM":
        # Good for high-dimensional data – scale first
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale",
                        random_state=seed))
        ])

    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed
        )

    if name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])

    if name == "ETREE":  # current default
        return ExtraTreesClassifier(
            n_estimators=200, max_depth=3, random_state=seed
        )

    if name == "LDA":
        # LDA benefits from scaling if features have very different variances
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis())
        ])

    raise ValueError(f"Unknown evaluation model: {name}")


import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

def evaluate_once(model, X_train, X_test, y_train, y_test,
                  fixed_threshold):
    """
    Train 'model', make hard predictions at 'fixed_threshold', and
    return a metrics dict.  Works for binary or multi-class.
    """

    # ------------- training ------------------------------
    model.fit(X_train, y_train)

    # ------------- probabilities -------------------------
    y_proba = model.predict_proba(X_test)
    n_classes = y_proba.shape[1]

    # ------------- hard predictions ----------------------
    if n_classes == 2:                                   # ---------- binary
        # Column indices ↔ class labels
        neg_label, pos_label = model.classes_            # e.g. (-2, 0)

        pos_idx = 1                                      # prob column for pos_label
        neg_idx = 0                                      # prob column for neg_label

        y_hat = np.where(
            y_proba[:, pos_idx] >= fixed_threshold,
            pos_label,                                   # predict positive
            neg_label                                    # else negative
        )

        auc = roc_auc_score(y_test, y_proba[:, pos_idx])

    else:                                                # ------- multi-class
        y_hat = model.classes_[y_proba.argmax(axis=1)]
        auc   = roc_auc_score(y_test, y_proba, multi_class="ovr")

    # ------------- metrics -------------------------------
    metrics = {
        "acc":  accuracy_score(y_test, y_hat),
        "auc":  auc,
        "prec": precision_score(y_test, y_hat,
                                average="weighted", zero_division=0),
        "rec":  recall_score(y_test, y_hat,
                            average="weighted", zero_division=0),
        "f1":   f1_score(y_test, y_hat,
                        average="weighted", zero_division=0),
    }

    # --------------------------------------------------
    # Metrics
    return {
        "acc":  accuracy_score(y_test, y_hat),
        "auc":  auc,
        "prec": precision_score(y_test, y_hat,
                                average="weighted", zero_division=0),
        "rec":  recall_score(y_test, y_hat,
                             average="weighted", zero_division=0),
        "f1":   f1_score(y_test, y_hat,
                         average="weighted", zero_division=0),
    }




# -------------------------------------------------------------------
# helper – keeps the long if/elif for FS exactly as you had
# -------------------------------------------------------------------
def run_fs(data_name,fs_fn, algo_name, X_tr, y_tr, X_te, y_te, k, row, device):
    """Dispatch all the existing FS cases, unchanged from your code."""
    if algo_name in ['DT', 'LeadingEV']:
        return fs_fn(X_tr, k, seed)
    elif algo_name in ['low_variance', 'CT', 'Skewness']:
        return fs_fn(X_tr, k, row, seed)
    elif algo_name in ['LS', 'MCFS']:
        return fs_fn(X_tr, len(np.unique(y_tr)), k, row, seed)
    elif algo_name in ['ReliefF']:
        return fs_fn(X_tr, y_tr, k, row, len(np.unique(y_tr)), seed)
    elif algo_name in ['CAE', 'UFS', 'DLFS', 'GRACES']:
        if algo_name == 'DLFS':
            return fs_fn(data_name, X_tr, y_tr, X_te, y_te,
                         k, row, seed, device)
        if algo_name == 'UFS':
            X_tr, X_te, y_tr, out = fs_fn(X_tr, y_tr, X_te, y_te,
                                          k, row, seed, device)
            return out
        return fs_fn(X_tr, y_tr, k, row, seed, device)
    else:
        return fs_fn(X_tr, y_tr, k, row, seed)


import numpy as np
from typing import Tuple
from sklearn.inspection import permutation_importance
from sklearn.ensemble   import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression






def calc_feature_importance_by_model(
    model_name,
    X,
    y,
    candidate_indices,      # 1-D array of K column indices
    seed):
    """
    Fast importance extractor.

    • Trees → .feature_importances_
    • LogisticRegression → |coef|
    • SVM / KNN / other → *surrogate* ExtraTrees (much faster than permutation)
    • Optional row subsampling for very large X to keep runtime bounded.
    """
    subsample_rows = None
    perm_repeats = 5
    # 0️⃣ optional subsample to speed things up on big data
    if subsample_rows is not None and X.shape[0] > subsample_rows:
        idx = np.random.RandomState(seed).choice(
            X.shape[0], subsample_rows, replace=False
        )
        X, y = X[idx], y[idx]

    # 1️⃣ slice to the K candidate cols
    X_sel = X[:, candidate_indices]

    # 2️⃣ fresh model identical to evaluation step
    model = get_eval_model(model_name, seed)
    model.fit(X_sel, y)

    # 3️⃣ raw importance
    if isinstance(model, (ExtraTreesClassifier, RandomForestClassifier)):
        raw_imp = model.feature_importances_

    elif isinstance(model, LogisticRegression):
        raw_imp = np.mean(np.abs(model.coef_), axis=0)

    elif model_name in {"SVM", "KNN"}:
        # ➡️ fallback: quick ExtraTrees surrogate
        surrogate = ExtraTreesClassifier(
            n_estimators=120, max_depth=3,
            random_state=seed, n_jobs=-1
        )
        surrogate.fit(X_sel, y)
        raw_imp = surrogate.feature_importances_

    else:
        # generic but slower → permutation on (small) subsample
        perm = permutation_importance(
            model, X_sel, y,
            n_repeats=perm_repeats,
            random_state=seed,
            n_jobs=-1
        )
        raw_imp = perm.importances_mean

    # 4️⃣ sort high→low
    order   = np.argsort(raw_imp)[::-1]
    return candidate_indices[order], raw_imp[order]



def flatten_eval_df(df_long) :
    wide = {}
    for _, r in df_long.iterrows():
        pref = r["Classifier Model"]
        for col, val in r.items():
            if col == "Classifier Model":
                continue
            wide[f"{pref}_{col}"] = (
                [round(float(x),3) for x in val] if isinstance(val, list)
                else round(float(val),3) if isinstance(val, (float, np.floating))
                else val
            )
    return pd.DataFrame([wide])


def run_cross_validation(fixed_threshold,repository, data_name, P, data_arr, label_arr, row, k, algo_name,hyperparamters,FS_HD, device):
    
    
    # Helper – duplicate rows to guarantee all classes appear in both sets
    # ---------------------------------------------------------------------------
    import numpy as np
    import time          # NEW – for wall-clock timing
    def ensure_class_presence(X_tr, y_tr, X_te, y_te):
        """
        If a class is absent from either train or test, copy ONE example of that
        class from the opposite side.  Returns (X_tr2, y_tr2, X_te2, y_te2).
        """
        classes = np.unique(np.concatenate([y_tr, y_te]))

        # --- copy missing classes into TRAIN ------------------------------------
        missing_in_train = [c for c in classes if c not in y_tr]
        if missing_in_train:
            # pick first occurrence of each missing class from test
            mask = np.isin(y_te, missing_in_train)
            X_tr  = np.vstack([X_tr,  X_te[mask]])
            y_tr  = np.concatenate([y_tr, y_te[mask]])

        # --- copy missing classes into TEST -------------------------------------
        missing_in_test = [c for c in classes if c not in y_te]
        if missing_in_test:
            # pick first occurrence of each missing class from train
            mask = np.isin(y_tr, missing_in_test)
            X_te = np.vstack([X_te, X_tr[mask]])
            y_te = np.concatenate([y_te, y_tr[mask]])

        return X_tr, y_tr, X_te, y_te

    # 

    
    """
    Perform cross-validation for a given dataset and algorithm, computing accuracy, precision,
    recall, F1, and AUC. It collects the selected K feature indices from each fold and returns
    a consensus set of best K indices based on frequency across folds.
    
    In addition to standard metrics, the function computes performance strings for Accuracy, AUC,
    Precision, Recall, and F1. It also returns a list of scores for each metric across folds and
    the maximum score observed for each metric.
    
    Importantly, it computes two different best accuracy values:
      - Best Accuracy (SHAP): Accuracy from the fold with the highest average SHAP importance.
      - Best Accuracy (Feature Importance): Accuracy from the fold with the highest average feature importance.
    
    Returns:
        tuple: (result_df, best_accuracy_shap, best_accuracy_fi, mean_auc, best_indices_shap_sorted, best_indices_fi_sorted)
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from collections import Counter
    import math
    import shap
    from sklearn.ensemble import ExtraTreesClassifier

    
    

    # ---------- helpers ---------------------------------------------------------
    def performance_label(x):
        return "High" if x > .9 else "Low" if x < .7 else "Fair"

    rows, columns_len     = data_arr.shape
    p_n_ratio             = round(int(P) / len(label_arr), 3)
    p_n_classification    = ("Below 1" if p_n_ratio < 1
                             else "Between 1 to 10" if p_n_ratio < 10
                             else "Above 10")
    num_classes           = len(np.unique(label_arr))

    # --------------------------------------------------

    # ---------- evaluation setup -----------------------------------------------
    from configs.config import evaluation_models

    EVAL_MODELS = evaluation_models
    metrics = {m: defaultdict(list) for m in EVAL_MODELS}          # {model: {metric: []}}
    eval_times = {m: [] for m in EVAL_MODELS}    # NEW
    
    # ---------- store fold-level objects for later FI calculation --------------
    folds_for_fi = []     # each element: (X_train, y_train, selected_feats)


    # ---------- CV loop --------------------------------------------------------
    start_time = datetime.now()
    successful = False
    cv_start_ts = time.time() 
    # try k = initial_splits, initial_splits-1, …, final_splits
    for n_splits in range(initial_splits, final_splits - 1, -1):
        try:
            skf = StratifiedKFold(n_splits=n_splits,
                                  shuffle=True, random_state=seed)

            for fold_idx, (tr, te) in enumerate(skf.split(data_arr, label_arr), 1):
                # ------------------------------------------------------------
                # Split & guarantee class presence on both sides
                # ------------------------------------------------------------
                X_train, X_test = data_arr[tr], data_arr[te]
                y_train, y_test = label_arr[tr], label_arr[te]

                X_train, y_train, X_test, y_test = ensure_class_presence(
                    X_train, y_train, X_test, y_test
                )

                # ------------------------------------------------------------
                # 1 · Feature Selection
                # ------------------------------------------------------------
                fs_fn = feature_selection_mapping[algo_name]
                selected = run_fs(data_name, fs_fn, algo_name,
                                  X_train, y_train, X_test, y_test,
                                  k, row, device)

                folds_for_fi.append((X_train, y_train, np.array(selected)))

                X_tr_sel = X_train[:, selected]
                X_te_sel = X_test[:,  selected]

                # ------------------------------------------------------------
                # 2 · Evaluate with each model
                # ------------------------------------------------------------
                for m_name in EVAL_MODELS:
                    t0 = time.time()
                    clf = get_eval_model(m_name, seed)
                    res = evaluate_once(clf, X_tr_sel, X_te_sel,
                                        y_train, y_test,
                                        fixed_threshold)
                    eval_times[m_name].append(time.time() - t0)   # NEW stop

                    for key, val in res.items():
                        metrics[m_name][key].append(val)

            successful = True
            break   # finished all folds for this n_splits

        except Exception as e:
            print(f"CV error ({n_splits} splits):", e)
            continue  # try a smaller n_splits

    if not successful:
        raise RuntimeError("Cross-validation failed for all split counts")

    #  Cross-validation timer
    cv_runtime_sec = time.time() - cv_start_ts 

    # ---------- 3. Pick evaluator with best mean-AUC ---------------------------
    mean_auc = {m: np.mean(d["auc"]) for m, d in metrics.items()}
    mean_acc = {m: np.mean(d["acc"]) for m, d in metrics.items()}

    best_eval = max(mean_auc, key=mean_auc.get)
    best_eval_Acc = max(mean_acc, key=mean_acc.get)


    #print(f"\n>>> Best evaluator by mean-AUC: {best_eval}  "
    #      f"(mean AUC = {mean_auc[best_eval]:.3f})")
    
    #print(f"\n>>> Best evaluator by mean-Accuracy: {best_eval_Acc}  "
    #      f"(mean Acc = {mean_acc[best_eval_Acc]:.3f})")
    

    # ------------------------------------------------------------------
    # 4. Re-rank features – do it once *per evaluator*
    # ------------------------------------------------------------------
    


    indices_per_eval = {}          # {model: {freq, avg_rank, mean_fi}}
    for m_name in EVAL_MODELS:
        fold_lists = []            # per-fold ranked-idx list
        fi_dict    = defaultdict(list)

        for X_tr, y_tr, sel_feats in folds_for_fi:
            idx_imp, imp_vals = calc_feature_importance_by_model(
                m_name, X_tr, y_tr, sel_feats, seed
            )
            fold_lists.append(idx_imp.tolist())
            for i, v in zip(idx_imp, imp_vals):
                fi_dict[i].append(v)

        indices_per_eval[m_name] = dict(
            freq      = consensus_frequency(fold_lists, k),
            avg_rank  = consensus_average_rank(fold_lists, k),
            mean_fi   = consensus_mean_importance(fi_dict, k)
        )

    # ------------------------------------------------------------------
    # 5. Build one summary row for every evaluator
    # ------------------------------------------------------------------
    def _r3(x): return round(float(x), 3)


    # Check if hyperparamters is a string
    if isinstance(hyperparamters, str):
        hyper_comment = hyperparamters  # Use the string directly
    else:
        # Safely access the 'comment' column
        if 'comment' in hyperparamters.columns and not hyperparamters['comment'].empty:
            hyper_comment = hyperparamters['comment'].iloc[0]
        else:
            hyper_comment = 'No_hyperparamters'  # Default value if 'comment' is missing or empty    

    # (a) constant metadata for all rows
    meta_common = {

        "Repository":            repository,
        "Dataset":               data_name,
        "Algorithm":             algo_name,
        "FS/HD":                 FS_HD,
        "N":                     len(label_arr),
        "P":                     int(P),
        "P/N":                   round(int(P) / len(label_arr), 3),
        "P/N Classification":    p_n_classification,
        "K":                     int(k),

        'K_%' :                  round(((k) / columns_len) * 100, 3),
        'Ln(K_%)' :              round(math.log(((k) / columns_len)), 3),
        "Hyperparamter":         str(hyper_comment),
        "Perform_CV[Y/N]":       perform_cv,
        "#Folds":                int(initial_splits),
        "#Successfuls_folds":    int(n_splits),
        "Probability Threshold": fixed_threshold,
        "Pipeline Runtime [Sec]": round(cv_runtime_sec, 2),
        "FS Time [Sec]": round((cv_runtime_sec/n_splits)-(np.mean(eval_times[m_name])), 2),
        "FS Time [Hr]": round(((cv_runtime_sec/n_splits)-(np.mean(eval_times[m_name])))/3600, 5), 
 

        
 


    }

    summary_rows = []
    for m_name, d in metrics.items():
        mu = {k: np.mean(v) for k, v in d.items()}
        sd = {k: np.std(v)  for k, v in d.items()}
        lists3 = {met: [_r3(x) for x in d[met]] for met in ["acc","auc","prec","rec","f1"]}
        idxset = indices_per_eval[m_name]

        # convert numpy ints → plain int
        idx_freq   = [int(i) for i in idxset["freq"]]
        idx_avg    = [int(i) for i in idxset["avg_rank"]]
        idx_meanfi = [int(i) for i in idxset["mean_fi"]]

        row = {
            **meta_common,                       # prepend metadata
            "Classifier Model":      m_name,
            # timing ---------------------------------------------------------------
            "Avg Classifier Time [Sec]": _r3(np.mean(eval_times[m_name])),   
            "Std_dev Classifier Time [Sec]": _r3(np.std(eval_times[m_name])),    

            'Data_Algo_rows_columns_k_Hyperparamter_Threshold_Evaluator_Key': 
            f"{data_name}_{algo_name}_{rows}_{columns_len}_{k}_{hyper_comment}_{fixed_threshold}_{m_name}",
            # mean / std metrics
            "Average Accuracy":      _r3(mu["acc"]),
            "Std_dev Accuracy":      _r3(sd["acc"]),
            "Average AUC":           _r3(mu["auc"]),
            "Std_dev AUC":           _r3(sd["auc"]),
            "Average Precision":     _r3(mu["prec"]),
            "Std_dev Precision":     _r3(sd["prec"]),
            "Average Recall":        _r3(mu["rec"]),
            "Std_dev Recall":        _r3(sd["rec"]),
            "Average F1":            _r3(mu["f1"]),
            "Std_dev F1":           _r3(sd["f1"]),


            # max metrics across folds for this evaluator
            "Max Accuracy (All Folds)":   _r3(max(d["acc"])),
            "Max AUC (All Folds)":        _r3(max(d["auc"])),
            "Max Precision (All Folds)":  _r3(max(d["prec"])),
            "Max Recall (All Folds)":     _r3(max(d["rec"])),
            "Max F1 (All Folds)":         _r3(max(d["f1"])),

            # per-fold metric lists (rounded)
            "Acc list":  lists3["acc"],
            "AUC list":  lists3["auc"],
            "Prec list": lists3["prec"],
            "Rec list":  lists3["rec"],
            "F1 list":   lists3["f1"],

            # consensus index lists
            "Best Indices (frequency)":      idx_freq,
            "Best Indices (average-rank)":   idx_avg,
            "Best Indices (mean-FI)":        idx_meanfi,
        }

        summary_rows.append(row)


    df_long = pd.DataFrame(summary_rows)
    df_wide = flatten_eval_df(df_long)

    return pd.DataFrame(df_long)






import math
def custom_round(x):
    """
    Rounds a float according to the following rules:
      - If abs(x) >= 0.001, round to 3 decimal places.
      - If abs(x) < 0.001 and nonzero, round so that only the first nonzero digit is preserved.
        For example, 0.0000005334 becomes 0.0000005.
      - Zero is returned as 0.0.
    """
    if x == 0:
        return 0.0
    abs_x = abs(x)
    if abs_x >= 0.001:
        return round(x, 3)
    else:
        exponent = math.floor(math.log10(abs_x))
        factor = 10 ** (-exponent)
        first_digit = int(abs_x * factor)
        result = first_digit / factor
        return math.copysign(result, x)





def create_shap_waterfall_plot(model, fold_test_data_selected, indices_for_selected, dataset_path):
    # Initialize the explainer with the model
    explainer = shap.Explainer(model)

    # Calculate SHAP values for the selected test data
    shap_values = explainer(fold_test_data_selected)

    # Select the first instance to plot as an example
    instance_index = 0  # Adjust as needed

    # Extract the SHAP values for a single instance
    single_shap_value = shap_values.values[instance_index]

    # Handling base_values depending on its structure
    if isinstance(shap_values.base_values, np.ndarray):
        if shap_values.base_values.ndim > 1:
            # Get a scalar by averaging base values if multiple values exist
            base_value = shap_values.base_values[instance_index].mean()
        else:
            # Directly use the scalar if appropriate
            base_value = shap_values.base_values[instance_index]
    else:
        # Use as is if it's already a scalar
        base_value = shap_values.base_values

    # Flatten the SHAP values if necessary (handling single or multi-feature)
    if single_shap_value.ndim > 1:
        single_shap_value = single_shap_value.flatten()

    # Create a SHAP Explanation object with a single instance's SHAP values
    single_explanation = shap.Explanation(values=single_shap_value,
                                        base_values=base_value,
                                        data=fold_test_data_selected[instance_index],
                                        feature_names=[str(idx) for idx in indices_for_selected])

    # Plot the SHAP waterfall plot for the selected instance
    plt.figure()
    shap.plots.waterfall(single_explanation, max_display=10, show=False)
    plt.title('SHAP Waterfall Plot for Instance ' + str(instance_index))

    # Save the plot
    shap_waterfall_path = os.path.join(dataset_path, f'shap_waterfall_instance_{instance_index}.png')
    plt.savefig(shap_waterfall_path)
    plt.close()

    print(f"SHAP Waterfall plot for instance {instance_index} saved to {shap_waterfall_path}")




                    
def run_train_test_split(specific_experiment, FS_HD, fold_train_data, fold_train_labels,
                         fold_test_data, fold_test_labels, data_name, row, k, P, repository,
                         algo_name, train_size, device=None):
    """
    Perform train-test split evaluation for a given dataset and algorithm, computing
    accuracy, precision, recall, F1, and AUC. In addition, it creates performance strings for
    Accuracy and AUC (as percentages with a performance label: High, Fair, Low) and returns the
    selected feature indices sorted by their SHAP importance (highest first).
    
    Args:
        specific_experiment (str): Identifier for the experiment.
        FS_HD (str): 'FS' or 'HD' mode.
        fold_train_data (np.ndarray): Training feature matrix.
        fold_train_labels (np.ndarray): Training labels.
        fold_test_data (np.ndarray): Testing feature matrix.
        fold_test_labels (np.ndarray): Testing labels.
        data_name (str): Name of the dataset.
        row (dict or str): Hyperparameter information (or a string for algorithms with no hyperparameters).
        k (int): Number of features to select.
        P (int): A value used for key generation (typically the number of features).
        repository (str): Repository name.
        algo_name (str): Name of the algorithm.
        train_size (float): Proportion of the dataset used for training.
        device (str, optional): Device to use for neural network methods (if applicable).
        
    Returns:
        tuple: (result_df, best_accuracy, best_AUC, best_indices_sorted)
            - result_df: One-row DataFrame with aggregated metrics.
            - best_accuracy: The accuracy on the test set.
            - best_AUC: The AUC on the test set.
            - best_indices_sorted: The selected feature indices sorted by their SHAP importance.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
    from sklearn.ensemble import ExtraTreesClassifier
    import shap

    # Concatenate train and test for overall stats (if needed)
    label_arr = np.concatenate((fold_train_labels, fold_test_labels))
    data_arr = np.concatenate((fold_train_data, fold_test_data))
    unique_values = np.unique(label_arr)
    num_classes = len(unique_values)
    total_rows, columns_len = data_arr.shape
    random_state = seed  # assuming seed is defined globally

    # Determine hyperparameter comment.
    if isinstance(row, str):
        hyperparamter = row
    else:
        hyperparamter = row.get('comment', '')
    
    # Feature Selection Step.
    feature_selection_function = feature_selection_mapping[algo_name]
    if algo_name in ['DT', 'LeadingEV']:
        selected_features = feature_selection_function(fold_train_data, k, random_state)
    elif algo_name in ['low_variance', 'CT', 'Skewness']:
        selected_features = feature_selection_function(fold_train_data, k, row, random_state)
    elif algo_name in ['LS', 'MCFS']:
        selected_features = feature_selection_function(fold_train_data, num_classes, k, row, random_state)
    elif algo_name in ['ReliefF']:
        selected_features = feature_selection_function(fold_train_data, fold_train_labels, k, row, num_classes, random_state)
    elif algo_name in ['ETree', 'SVC', 'AdaBoost','SHAP', 'Univariate', 'alpha_investing',
                        'DecisionTree_Forward', 'SVM_Forward','SVM_RBF', 'GRAPH', 'RFS', 'mRMR',
                        'GRACES', 'DecisionTree_Backward', 'SVM_Backward']:
        selected_features = feature_selection_function(fold_train_data, fold_train_labels, k, row, random_state)
    elif algo_name in ['DLFS']:
        selected_features = feature_selection_function(data_name, fold_train_data, fold_train_labels,
                                                       fold_test_data, fold_test_labels, k, row, random_state)
    elif algo_name in ['CAE']:
        selected_features = feature_selection_function(fold_train_data, fold_test_data, k, row, random_state)
    elif algo_name in ['UFS', 'CARTE']:
        fold_train_data, fold_test_data, fold_train_labels, selected_features = \
            feature_selection_function(fold_train_data, fold_train_labels, fold_test_data, fold_test_labels, k, row, random_state)
    
    # Compute feature importance and get selected indices.
    indices_for_selected, feature_importance, accumulated_importance = \
        calculate_feature_importance_rf(fold_train_data, fold_train_labels, selected_features, random_state)
    best_indices = indices_for_selected  # Initial ordering.
    
    # Select features based on these indices.
    fold_train_data_selected = np.take(fold_train_data, indices_for_selected, axis=1)
    fold_test_data_selected = np.take(fold_test_data, indices_for_selected, axis=1)
    
    # Train model.
    clf = ExtraTreesClassifier(n_estimators=200, max_depth=3, random_state=random_state)
    clf.fit(fold_train_data_selected, fold_train_labels)
    y_pred = clf.predict(fold_test_data_selected)
    
    # Compute AUC.
    if num_classes == 2:
        y_proba = clf.predict_proba(fold_test_data_selected)[:, 1]
        auc = roc_auc_score(fold_test_labels, y_proba)
    else:
        y_proba = clf.predict_proba(fold_test_data_selected)
        auc = roc_auc_score(fold_test_labels, y_proba, multi_class='ovr')
    
    # Compute metrics.
    accuracy = accuracy_score(fold_test_labels, y_pred)
    prec = precision_score(fold_test_labels, y_pred, average='weighted', zero_division=0)
    rec = recall_score(fold_test_labels, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(fold_test_labels, y_pred, average='weighted', zero_division=0)
    
    # Compute SHAP values for the selected features on test data.
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(fold_test_data_selected)
    if num_classes == 2:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    else:
        # For multi-class, average the absolute values across classes.
        mean_abs_shap = np.mean([np.mean(np.abs(shap_class), axis=0) for shap_class in shap_values], axis=0)
    
    # Sort the selected indices based on the SHAP importance (descending order).
    sorted_order = np.argsort(mean_abs_shap)[::-1]
    best_indices_sorted = [best_indices[i] for i in sorted_order]
    
    # Since it's a single train-test split, standard deviations are zero.
    std_acc = 0.0
    std_auc = 0.0
    
    # Create performance labels.
    if accuracy > 0.9:
        acc_perf = "High"
    elif accuracy < 0.7:
        acc_perf = "Low"
    else:
        acc_perf = "Fair"
    acc_perf_str = f"{accuracy*100:.1f}% ± {std_acc*100:.1f}% ({acc_perf})"
    
    if auc > 0.9:
        auc_perf = "High"
    elif auc < 0.7:
        auc_perf = "Low"
    else:
        auc_perf = "Fair"
    auc_perf_str = f"{auc*100:.1f}% ± {std_auc*100:.1f}% ({auc_perf})"
    
    # Build results dictionary.
    results = {
        'Data_Algo_rows_columns_k_Hyperparamter_Threshold_Evaluator_Key': f"{data_name}_{algo_name}_{total_rows}_{columns_len}_{k}_{hyperparamter}",
        'Repository': repository,
        'Dataset': data_name,
        'Algorithm': algo_name,
        'N': len(label_arr),
        'P': int(P),
        'P/N': round(int(P) / len(label_arr), 3),
        "P/N Classification": ("Below 1" if round(int(P) / len(label_arr), 3) < 1 
                                 else "Between 1 to 10" if round(int(P) / len(label_arr), 3) < 10 
                                 else "Above 10"),
        'K': int(k),
        'Hyperparamter': hyperparamter,
        '#Folds': 'No_CV',
        '#successfuls_folds': 'No_CV',
        # Accuracy scores.
        'Average Accuracy': accuracy,
        'Std_dev_Accuracy': std_acc,
        'Acc ± Std (Performance)': acc_perf_str,
        'Max_Accuracy': accuracy,
        'Accuracies': [accuracy],
        'Best_accuracy[Dataset/K&Hyperparamters]': accuracy,
        'Best_indices_Sorted_Acc[Dataset/K&Hyperparamters]': best_indices_sorted,
        'Best_Feature_Imprtance_Acc[Dataset/K&Hyperparamters]': feature_importance if isinstance(feature_importance, list) else [],
        'Best_accumulated_importance_Accuracy': accumulated_importance,
        # AUC scores.
        "Average_auc_ovo": auc,
        "Std_dev_auc_ovo": std_auc,
        'AUC ± Std_ovo': f"{auc:.3f} ± {std_auc:.3f}",
        "Max_auc_ovo": auc,
        'AUC_scores_ovo': [auc],
        'Best_AUC_ovo[Dataset/K&Hyperparamters]': auc,
        'Best_indices_Sorted_AUC_ovo[Dataset/K&Hyperparamters]': best_indices_sorted,
        'Best_Feature_Imprtance_AUC_ovo[Dataset/K&Hyperparamters]': feature_importance if isinstance(feature_importance, list) else [],
        'Best_accumulated_importance_AUC_ovo': accumulated_importance,
        # For multi-class, OVR.
        "Average AUC": auc,
        "Std_dev_auc_ovr": std_auc,
        'AUC ± Std_ovr': auc_perf_str,
        "Max_auc_ovr": auc,
        'AUC_scores_ovr': [auc],
        'Best_AUC_ovr[Dataset/K&Hyperparamters]': auc,
        'Best_indices_Sorted_AUC_ovr[Dataset/K&Hyperparamters]': best_indices_sorted,
        'Best_Feature_Imprtance_AUC_ovr[Dataset/K&Hyperparamters]': feature_importance if isinstance(feature_importance, list) else [],
        'Best_accumulated_importance_AUC_ovr': accumulated_importance,
        'Running_time_5CV': "N/A",
        'RunningTime_5CV[Sec]': 0
    }
    
    result_df = pd.DataFrame([results])
    return result_df





def get_original_FS_AUC(dataset_path,no_hyperparameters,train_data_downsampled,train_labels_downsampled,test_data,test_labels,train_data_source,train_labels_source,auc_score_HD_Original,path, P, repository, data_name, seed,Hard_T,Training_size):


    # Initialize the results DataFrame
    results_df_All_FS_loop = pd.DataFrame()


    # Loop through the algorithms
    for algo_name, alg in feature_selection_mapping.items():
        hyperparamters = hyperparameters_mapping[algo_name]
        results = run_parallel_processing(
            alg, algo_name, K_List, dataset_path, hyperparamters, repository, data_name, P, initial_splits, final_splits, no_hyperparameters, perform_cv, train_size, 'FS'
        )

        df = pd.concat(results, ignore_index=True)

        if df.empty:
            print(f"The list contains empty DataFrames for {algo_name} - error in the code.")
        else:
            results_df_All_FS_loop = pd.concat([results_df_All_FS_loop, df], ignore_index=True)


    # Additional processing and return statement can be added here as needed
    file_path_excel = path + '/Main/Output/Convert Data 2 Hard/Change Log/df_original_trianing_AUC.xlsx'
    results_df_All_FS_loop['HD_ETREE'] = round(auc_score_HD_Original, 2)
    results_df_All_FS_loop['FS_Contribution'] = results_df_All_FS_loop['Average_auc_ovo'] - auc_score_HD_Original

    results_df_All_FS_loop['FS_Can_Help'] = np.where(results_df_All_FS_loop['Average_auc_ovo'] > auc_score_HD_Original, 'Yes', 'No')
    results_df_All_FS_loop['Hard_T'] = Hard_T
    results_df_All_FS_loop['Training_Set%'] = Training_size

    # Initialize an empty DataFrame
    empty_df = pd.DataFrame()

    for i in range(0, len(results_df_All_FS_loop)):
        df_temp = results_df_All_FS_loop.iloc[i]  # Use iloc to access rows by index
        indices_from_original_training = df_temp['Best_indices_Sorted_AUC_ovr[Dataset/K&Hyperparamters]']

        # Select the top features from train_data_new and test_data
        train_data_selected = np.take(train_data_downsampled, indices_from_original_training, axis=1)
        test_data_selected = np.take(test_data, indices_from_original_training, axis=1)

        # Train model with current hyperparameters
        clf = ExtraTreesClassifier(n_estimators=200, max_depth=3, random_state=seed)  # 200 & 3 shown as the BIC hyperparameter and also it's the default in scikit-learn
        clf.fit(train_data_selected, train_labels_downsampled)

        # Evaluate the model
        unique_values = np.unique(train_labels_downsampled)

        if len(unique_values) == 2:
            test_predicted_probs = clf.predict_proba(test_data_selected)[:, 1]
            test_auc_ovo = round(roc_auc_score(test_labels, test_predicted_probs), 2)
            test_auc_ovr = test_auc_ovo
        else:
            test_predicted_probs = clf.predict_proba(test_data_selected)
            test_auc_ovo = round(roc_auc_score(test_labels, test_predicted_probs, multi_class='ovo'), 2)
            test_auc_ovr = round(roc_auc_score(test_labels, test_predicted_probs, multi_class='ovr'), 2)

        print(f"Test AUC OVO: {test_auc_ovo}")
        print(f"Test AUC OVR: {test_auc_ovr}")
        print(f"Test AUC Without FS: {auc_score_HD_Original}")

        results_df_All_FS_loop.loc[i, 'FS_AUC_Downsampling with FS Training set'] = test_auc_ovo
        results_df_All_FS_loop['FS_AUC_30% Training'] = results_df_All_FS_loop['Average_auc_ovo']

        if i == 0:
            combined_data = results_df_All_FS_loop.drop_duplicates(subset=['Repository', 'Dataset', 'Hard_T', 'Training_Set%', 'Algorithm', 'K'], keep='last').sort_values(by=['Repository', 'Dataset', 'Hard_T', 'Training_Set%', 'Algorithm', 'K'], ascending=[True, True, True, True, True, True]).reset_index(drop=True)
        else:
            combined_data = pd.concat([combined_data, results_df_All_FS_loop]) \
                .drop_duplicates(subset=['Repository', 'Dataset', 'Hard_T', 'Training_Set%', 'Algorithm', 'K'], keep='last') \
                .sort_values(by=['Repository', 'Dataset', 'Hard_T', 'Training_Set%', 'Algorithm', 'K'], ascending=[True, True, True, True, True, True]) \
                .reset_index(drop=True)

    # Step 1: Find the row with the maximum test_auc_ovo
    max_test_auc_ovo = combined_data['FS_AUC_Downsampling with FS Training set'].max()  # FS_AUC_Downsampling with FS Training set
    row_with_max_test_auc_ovo = combined_data[combined_data['FS_AUC_Downsampling with FS Training set'] == max_test_auc_ovo]

    # Step 2: Get the value of Average_auc_ovo from this row
    max_average_auc_ovo_from_max_test_auc_ovo = row_with_max_test_auc_ovo['Average_auc_ovo'].max()

    # Print the results
    print("Row with max test_auc_ovo:")
    print(row_with_max_test_auc_ovo)
    print("\nMax Average_auc_ovo from the row with max test_auc_ovo:")
    print(max_average_auc_ovo_from_max_test_auc_ovo)  # FS_AUC_Downsampling with FS Training set

    if os.path.exists(file_path_excel):
        existing_data = pd.read_excel(file_path_excel)
        combined_data_all = pd.concat([existing_data, combined_data]) \
            .drop_duplicates(subset=['Repository', 'Dataset', 'Hard_T', 'Training_Set%', 'Algorithm', 'K'], keep='last') \
            .sort_values(by=['Repository', 'Dataset', 'Hard_T', 'Training_Set%', 'Algorithm', 'K'], ascending=[True, True, True, True, True, True]) \
            .reset_index(drop=True)
    else:
        combined_data_all = combined_data.reset_index(drop=True)

    # Save the combined data
    combined_data_all.to_excel(file_path_excel, index=False)

    return max_test_auc_ovo, max_average_auc_ovo_from_max_test_auc_ovo





def check_initial_split(data_arr, label_arr, initial_train_size):
    """
    Ensure the training size is at least 0.3, the test size is greater than 0.3,
    the training set has more than 50 rows, and the test set has more than 50 rows.

    This function addresses two key statistical considerations:
    1. Statistical Representativeness:
       50 Instances: Having at least 50 instances in your test set can help ensure that the test set
       is statistically representative of the underlying distribution of the data. This number helps
       in achieving a balance where the test results can be considered reliable for inferencing about
       the model's performance on unseen data.
    2. Central Limit Theorem (CLT):
       The Central Limit Theorem suggests that with a sufficiently large sample size, the distribution
       of the sample means will approximate a normal distribution, regardless of the shape of the
       population distribution. Typically, sample sizes of 30 or more are considered sufficient for
       the CLT to hold, but aiming for 50 gives a higher confidence in the robustness of this approximation,
       especially in diverse datasets.

    Args:
        data_arr (numpy.ndarray): The data array.
        label_arr (numpy.ndarray): The label array.
        initial_train_size (float): The initial training size percentage.

    Returns:
        float: A valid training size percentage.
    """
    num_samples = len(data_arr)
    train_size = initial_train_size

    # Calculate the minimum valid test size
    min_test_size = max(0.3, 50 / num_samples)

    max_attempts = 100  # Maximum number of attempts
    attempts = 0  # Initialize the counter

    while True:
        train_rows = int(train_size * num_samples)
        test_rows = num_samples - train_rows

        if train_size >= 0.3 and (1 - train_size) >= min_test_size and train_rows >= 50 and test_rows >= 50:
            break
        else:
            if train_rows < 50:
                train_size = 50 / num_samples  # Increase the training size to ensure at least 50 rows
            elif test_rows < 50:
                train_size = 1 - (50 / num_samples)  # Adjust the training size to ensure at least 50 rows in the test set
            else:
                train_size -= 0.01  # Decrease the training size slightly to meet the conditions

        attempts += 1  # Increment the counter

        if attempts >= max_attempts:
            print("Maximum attempts reached. Returning default training size of 0.3.")
            return 0.3  # Return default training size

        #if train_size < 0.3:
        #    raise ValueError("Cannot find a valid training size that meets the conditions.")

    return train_size



def fill_missing_labels(train_data, test_data, train_labels, test_labels):
    """
    Fill missing labels in the training and testing sets.

    Args:
    - train_data (np.ndarray): Training data.
    - test_data (np.ndarray): Test data.
    - train_labels (np.ndarray): Training labels.
    - test_labels (np.ndarray): Test labels.

    Returns:
    - tuple: Updated training data, test data, training labels, and test labels.
    """
    label_arr = np.concatenate((test_labels, train_labels))
    unique_values = np.unique(test_labels)

    # Concatenate the labels back into label_arr
    label_arr = np.concatenate((train_labels, test_labels))
    data_arr = np.concatenate((train_data, test_data))

    n = len(label_arr)
    P = train_data.shape[1]
    k = P

    results_dataset = analyze_labels(label_arr)

    results_df_All_RF_loop = pd.DataFrame()

    # Identify missing classes in the training set
    all_unique_classes = np.unique(train_labels)
    unique_train_classes = np.unique(train_labels)
    unique_test_classes = np.unique(train_labels)  # Assuming test_labels is similar to train_labels for this example
    missing_classes_train = np.setdiff1d(all_unique_classes, unique_train_classes)
    missing_classes_test = np.setdiff1d(all_unique_classes, unique_test_classes)

    # Add missing classes to the training set
    for missing_class in missing_classes_train:
        missing_class_indices = np.where(train_labels == missing_class)[0]
        if len(missing_class_indices) > 0:
            train_data = np.vstack([train_data, train_data[missing_class_indices]])
            train_labels = np.hstack([train_labels, train_labels[missing_class_indices]])

    # Add missing classes to the test set
    for missing_class in missing_classes_test:
        missing_class_indices = np.where(train_labels == missing_class)[0]
        if len(missing_class_indices) > 0:
            test_data = np.vstack([train_data, train_data[missing_class_indices]])
            test_labels = np.hstack([train_labels, train_labels[missing_class_indices]])

    # Ensure each class has at least 2 members
    for cls in np.unique(train_labels):
        class_indices = np.where(train_labels == cls)[0]
        while len(class_indices) < 2:
            train_data = np.vstack([train_data, train_data[class_indices]])
            train_labels = np.hstack([train_labels, train_labels[class_indices]])
            class_indices = np.where(train_labels == cls)[0]

    return train_data, test_data, train_labels, test_labels

def HD_ETREE(results_dataset,D_Counter, train_data, test_data, train_labels, test_labels, repository, data_name, seed):

    print('Dataset_Reduction Phase: \033[4m' + str(round(D_Counter, 2) * 10) + '%\033[0m;')

    P = train_data.shape[1]
    k = P

    # Import specific functions from algorithms
    from fs_algorithms.AdaBoost import AdaBoost_FS
    from fs_algorithms.ETree import ETree_FS

    # Dictionary to map algorithm names to their respective functions
    random_forest_algorithms = {
        'AdaBoost': AdaBoost_FS,
        'ETree': ETree_FS,
    }

    # Define a dictionary to map algorithm names to their corresponding hyperparameters
    random_forest_hyperparameters = {
        'AdaBoost': AdaBoost_hyper,
        'ETree': ETree_hyper,
    }



    df = pd.DataFrame()
    results_df_All_RF_loop = pd.DataFrame()

    # Graces requried positive labels only
    #if algo_name in ['GRACES'] and np.any(label_arr < 0):
    #    # Convert labels to running numbers
    #    label_arr = convert_labels_to_running_numbers(label_arr)


    # Loop through the Random Forest algorithms
    for algo_name, alg in random_forest_algorithms.items():
        hyperparamters = random_forest_hyperparameters[algo_name]
        # Extract the row using iloc
        hyperparamters = hyperparamters.iloc[0]

        df = run_train_test_split(train_data, train_labels,test_data,test_labels, data_name, hyperparamters, k, P, repository, algo_name, train_size)


        #df = alg(train_data, train_labels, k, hyperparamters, seed)
        
        df['Majority_class_percentage'] = round(results_dataset['majority_class_percentage'] / 100, 2)
        df['#Classes'] = results_dataset['total_unique_labels']
        df['Class_Fraction'] = round(results_dataset['class_fraction'], 2)
        df['Average AUC'].max(), df, algo_name, "HD"

        if df.empty:
            print(f"The list contains empty DataFrames for {algo_name} - error in the code.")
        else:
            results_df_All_RF_loop = pd.concat([results_df_All_RF_loop, df], ignore_index=True)

    # Find the index of the row with the maximum value in 'Average AUC'
    max_index_auc = results_df_All_RF_loop['Average AUC'].idxmax()
    # Select the row with the maximum value and convert to DataFrame
    max_auc_ovr_rows = results_df_All_RF_loop.loc[max_index_auc].to_frame().transpose()

    # Get the algorithms for 'Average Accuracy'
    algorithms_auc = ','.join(max_auc_ovr_rows['Algorithm'].unique())

    max_index_acc = results_df_All_RF_loop['Average Accuracy'].idxmax()
    # Select the row with the maximum value and convert to DataFrame
    max_accuracy_rows = results_df_All_RF_loop.loc[max_index_acc].to_frame().transpose()

    # Get the algorithms for 'Average Accuracy'
    algorithms_accuracy = ','.join(max_accuracy_rows['Algorithm'].unique())
    return max_auc_ovr_rows['Average AUC'].max(),max_auc_ovr_rows,max_accuracy_rows['Average Accuracy'].max(),max_accuracy_rows, algorithms_auc,algorithms_accuracy,results_df_All_RF_loop


def process_input_Convert_Hard(k, P,train_data, train_labels,test_data,test_labels, data_name, hyperparamters, repository, algo_name, train_size,no_hyperparameters):
    
    df = pd.DataFrame()
    #if algo_name in no_hyperparameters:
    result_df, best_accuracy, best_AUC_ovo, best_AUC_ovr = run_train_test_split(train_data, train_labels,test_data,test_labels, data_name, hyperparamters, k, P, repository, algo_name, train_size)
    df = pd.concat([result_df, df], ignore_index=True)
    return df



def FS_Algorithms(results_dataset,D_Counter, train_data, test_data, train_labels, test_labels, repository, data_name,LowT,no_hyperparameters,K_List,results_df_All_FS_loop,AUC_FS,last_FS_RUN):
    
    """
    Perform feature selection algorithms and return the results.

    Args:
    - flag_FS_Higher_LowT (bool): Flag indicating whether to use higher or lower threshold.
    - D_Counter (int): Counter for dataset removal.
    - train_data (np.ndarray): Training data.
    - test_data (np.ndarray): Test data.
    - train_labels (np.ndarray): Training labels.
    - test_labels (np.ndarray): Test labels.
    - repository (str): Repository name.
    - data_name (str): Dataset name.
    - LowT (float): Low threshold value.
    - hyperparameters_mapping (dict): Mapping of algorithm names to their hyperparameters.
    - K_List (list): List of K values.
    - results_df_All_FS_loop (pd.DataFrame): DataFrame containing results for all FS loops.

    Returns:
    - tuple: Results of the feature selection algorithms.
    """

    print('Dataset_Reduction Phase: \033[4m' + str(round(D_Counter, 2) * 10) + '%\033[0m;')
    P = train_data.shape[1]
    k = P

    Counter_FS_Algorithm = 1  # Initialize the counter

    # Loop through the FS algorithms starting from item 20
    for index, (algo_name, alg) in enumerate(feature_selection_mapping.items()):
        if index > 20:  # Skip the last 4 algorithms (0-based index)
            continue
        try:
            print(f"Algorithm {Counter_FS_Algorithm}: {algo_name}")

            hyperparamters = hyperparameters_mapping[algo_name]
            
            if isinstance(hyperparamters, str):
                print('No Hyperparamters')
            else:
                # Extract the row using iloc
                hyperparamters = hyperparamters.iloc[0]

            # Graces required positive labels only
            if algo_name in ['GRACES'] and np.any(train_labels < 0):
                # Convert labels to running numbers
                train_labels = convert_labels_to_running_numbers(train_labels)
                test_labels = convert_labels_to_running_numbers(test_labels)

            # Run Parallel Processing through K_List in Config.py
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(process_input_Convert_Hard)(
                k, P, train_data, train_labels, test_data, test_labels, data_name, hyperparamters, repository, algo_name, train_size, no_hyperparameters
            ) for k in K_List)
            df = pd.concat(results, ignore_index=True)

            # Additional processing if needed
            df['Majority_class_percentage'] = round(results_dataset['majority_class_percentage'] / 100, 2)
            df['#Classes'] = results_dataset['total_unique_labels']
            df['Class_Fraction'] = round(results_dataset['class_fraction'], 2)
            df['Average AUC'].max(), df, algo_name, "HD"

        except ValueError as e:
            print(f"An error occurred with algorithm {algo_name}: {e}")
            continue  # Continue to the next algorithm

        Counter_FS_Algorithm += 1  # Increment the counter

        if df.empty:
            print(f"The list contains empty DataFrames for {algo_name} - error in the code.")
        else:
            results_df_All_FS_loop = pd.concat([results_df_All_FS_loop, df], ignore_index=True)
            
        print(f"Algorithm {Counter_FS_Algorithm}: {algo_name}, Max Average AUC: {df['Average AUC'].max()}")

        if df['Average AUC'].max() >=LowT :
            flag_FS_Higher_LowT = True
            if df['Average AUC'].max() >= AUC_FS : # in this case the performance of peeling dataset >Original dataset
                break  
            if len(train_labels) > 150 or  D_Counter< 9 : # we have anohter step of peeling so break it earlier and try to find harder dataset
                last_FS_RUN ='Y'
                break
 
        if Counter_FS_Algorithm == 21: # Don't include Wrapper backword
            break



    # Find the index of the row with the maximum value in 'Average AUC'
    max_index_auc = results_df_All_FS_loop['Average AUC'].idxmax()
    # Select the row with the maximum value and convert to DataFrame
    max_auc_ovr_rows = results_df_All_FS_loop.loc[max_index_auc].to_frame().transpose()

    # Get the algorithms for 'Average Accuracy'
    algorithms_auc = ','.join(max_auc_ovr_rows['Algorithm'].unique())

    max_index_acc = results_df_All_FS_loop['Average Accuracy'].idxmax()
    # Select the row with the maximum value and convert to DataFrame
    max_accuracy_rows = results_df_All_FS_loop.loc[max_index_acc].to_frame().transpose()

    # Get the algorithms for 'Average Accuracy'
    algorithms_accuracy = ','.join(max_accuracy_rows['Algorithm'].unique())
    return max_auc_ovr_rows['Average AUC'].max(),max_auc_ovr_rows,max_accuracy_rows['Average Accuracy'].max(),max_accuracy_rows, algorithms_auc,algorithms_accuracy,results_df_All_FS_loop,last_FS_RUN



# Function to save results
def save_results(core_number,train_labels, test_labels, test_data, D_Counter, FS_HD, algorithms_auc, algorithms_accuracy, max_average_auc_ovr, max_average_accuracy, max_auc_ovr_rows, LowT, Training_size, rows, results_df_All_RF_loop):
    """
    Save the results of the experiment to an Excel file.

    Args:
    - train_labels (np.ndarray): Training labels.
    - test_labels (np.ndarray): Test labels.
    - test_data (np.ndarray): Test data.
    - D_Counter (int): Counter for dataset removal.
    - FS_HD (str): Indicates whether the experiment is FS or HD.
    - algorithms_auc (list): List of AUC values for algorithms.
    - algorithms_accuracy (list): List of accuracy values for algorithms.
    - max_average_auc_ovr (float): Maximum average AUC OVR.
    - max_average_accuracy (float): Maximum average accuracy.
    - max_auc_ovr_rows (pd.DataFrame): DataFrame containing max AUC OVR rows.
    - LowT (float): Low threshold value.
    - Training_size (float): Training set size percentage.
    - rows (int): Number of rows in the dataset.
    - results_df_All_RF_loop (pd.DataFrame): DataFrame containing results for all RF loops.

    Returns:
    - pd.DataFrame: The updated DataFrame containing experiment details and feature robustness metrics.
    """
    results_df_All = pd.DataFrame(columns=["Dataset", "Data"])

    # Calculate the percentage of dataset removal
    data_removal = round(D_Counter / 10, 2)

    # Get the number of training and test labels
    len_train_labels = len(train_labels)
    len_test_labels = len(test_labels)

    # Define variables for the new row
    len_train_labels_phase = len_train_labels

    # Get the number of columns in the test data
    num_columns = test_data.shape[1]

    # Process AUC scores
    unique_values = max_auc_ovr_rows['Average_auc_ovr'].astype(str)
    highest_value = max(unique_values, key=lambda x: float(x))
    bold_highest_value = f"<b>{highest_value}</b>"
    unique_values_bolded = [bold_highest_value if value == highest_value else value for value in unique_values]
    auc_scores = "','".join(unique_values_bolded)
    auc_scores = "'" + auc_scores + "'"

    # Process Accuracy scores
    unique_values_acc = max_auc_ovr_rows['Average Accuracy'].astype(str)
    highest_value_acc = max(unique_values_acc, key=lambda x: float(x))
    bold_highest_value_acc = f"<b>{highest_value_acc}</b>"
    unique_values_bolded_acc = [bold_highest_value_acc if value == highest_value_acc else value for value in unique_values_acc]
    acc_scores = "','".join(unique_values_bolded_acc)
    acc_scores = "'" + acc_scores + "'"

    max_auc_ovr_rows.reset_index(drop=True, inplace=True)

    # Create a new row for the results DataFrame
    new_row = pd.DataFrame({
        "Repository": max_auc_ovr_rows['Repository'][0],
        "Dataset": [max_auc_ovr_rows['Dataset'][0]],
        "Hard_T": [LowT],
        "Training_Set%": [Training_size],
        "N_Source": [rows],
        "# Features": [num_columns], 
        
        # Until here this is the key for the original experiment
        "FS/HD": [FS_HD],
        "k": [test_data.shape[1]], # Number of Features in dataset    
        "% of Dataset removal": [data_removal],
        "Training Length": [len_train_labels_phase],
        "Test Length": [len_test_labels],

        "Leading_Algorithms_AUC": [algorithms_auc],
        "Leading_Algorithms_Acc": [algorithms_accuracy],
        "AUC_Score": [max_average_auc_ovr],
        "Accuracy_Score": [max_average_accuracy],
        "AUC_Scores(K=1/5/10/30/60)": [auc_scores],
        "Accuracy_Scores(K=1/5/10/30/60)": [acc_scores],
        "DataFrame": [max_auc_ovr_rows]
    })

    # List of algorithm columns to add
    algorithm_columns = [
        "ETree", "AdaBoost","SHAP", "Univariate", "low_variance", "SVC","SVM_RBF", "LS", "Skewness", "MCFS", "DT", "ReliefF", "alpha_investing",
        "DecisionTree_Forward", "DLFS", "SVM_Forward", "GRAPH", "RFS", "LeadingEV", "CAE", "mRMR", "CT", "GRACES", "UFS",
        "DecisionTree_Backward", "SVM_Backward","CEI_GA"
    ]

    # Add algorithm columns to the new row
    for algorithm in algorithm_columns:
        if FS_HD == 'HD' and algorithm in ["ETree", "AdaBoost"]:
            # Extract the value from results_df_All_RF_loop
            algorithm_value = results_df_All_RF_loop.loc[results_df_All_RF_loop['Algorithm'] == algorithm, 'AUC ± Std_ovr'].values
            if len(algorithm_value) > 0:
                new_row[algorithm] = algorithm_value[0]
            else:
                new_row[algorithm] = None
        elif FS_HD != 'HD' and algorithm not in ["ETree", "AdaBoost"]:
            # Extract the value from results_df_All_RF_loop
            algorithm_value = results_df_All_RF_loop.loc[results_df_All_RF_loop['Algorithm'] == algorithm, 'AUC ± Std_ovr'].values
            if len(algorithm_value) > 0:
                new_row[algorithm] = algorithm_value[0]
            else:
                new_row[algorithm] = None
        else:
            new_row[algorithm] = None

    # Convert the DataFrame column to JSON
    new_row['DataFrame'] = new_row['DataFrame'].apply(lambda df: df.to_json())

    # Combine the new row with the existing results DataFrame
    if results_df_All.empty or results_df_All.isna().all().all():
        results_df_All = new_row
        results_df_All_Short = results_df_All.copy()
    else:
        combined_data = pd.concat([results_df_All, new_row]) \
                          .drop_duplicates(subset=['Repository', 'Dataset', 'Training Length', 'FS/HD', 'Algorithm', 'Hard_T', 'Training_Set%'], keep='last') \
                          .reset_index(drop=True)

        # Sort the data based on 'Repository', 'Dataset' in ascending order and 'FS/HD', 'Training Length' in descending order
        combined_data = combined_data.sort_values(by=['Repository', 'Dataset', 'FS/HD', 'Training Length'], ascending=[True, True, False, True])
        results_df_All_Short = combined_data.copy()

    # Drop the 'DataFrame' column from the short results DataFrame
    results_df_All_Short.drop('DataFrame', axis=1, inplace=True)

    # Define the file path for the AUC Excel file
    file_path_excel_auc = os.path.join(convert_data_hard_path_Source, 'Change Log/'+core_number+'df_change_log_AUC.xlsx')

    # Save the combined data to the AUC Excel file
    if os.path.exists(file_path_excel_auc):
        existing_data_auc = pd.read_excel(file_path_excel_auc)
        combined_data_auc = pd.concat([existing_data_auc, results_df_All_Short]) \
                            .drop_duplicates(subset=['Repository', 'Dataset', '% of Dataset removal', 'FS/HD', 'Hard_T', 'Training_Set%'], keep='last') \
                            .sort_values(by=['Repository', 'Dataset', 'Training Length', 'FS/HD'], ascending=[True, True, False, True]) \
                            .reset_index(drop=True)
    else:
        combined_data_auc = results_df_All_Short.reset_index(drop=True)

    combined_data_auc.to_excel(file_path_excel_auc, index=False)

    # Add algorithm columns to the new row for Accuracy
    for algorithm in algorithm_columns:
        if FS_HD == 'HD' and algorithm in ["ETree", "AdaBoost"]:
            # Extract the value from results_df_All_RF_loop
            algorithm_value = results_df_All_RF_loop.loc[results_df_All_RF_loop['Algorithm'] == algorithm, 'Acc ± Std'].values
            if len(algorithm_value) > 0:
                new_row[algorithm] = algorithm_value[0]
            else:
                new_row[algorithm] = None
        elif FS_HD != 'HD' and algorithm not in ["ETree", "AdaBoost"]:
            # Extract the value from results_df_All_RF_loop
            algorithm_value = results_df_All_RF_loop.loc[results_df_All_RF_loop['Algorithm'] == algorithm, 'Acc ± Std'].values
            if len(algorithm_value) > 0:
                new_row[algorithm] = algorithm_value[0]
            else:
                new_row[algorithm] = None
        else:
            new_row[algorithm] = None

    # Define the file path for the Accuracy Excel file
    file_path_excel_accuracy = os.path.join(convert_data_hard_path_Source, 'Change Log/'+core_number+'df_change_log_Accuracy.xlsx')

    # Save the combined data to the Accuracy Excel file
    if os.path.exists(file_path_excel_accuracy):
        existing_data_accuracy = pd.read_excel(file_path_excel_accuracy)
        combined_data_accuracy = pd.concat([existing_data_accuracy, results_df_All_Short]) \
                                .drop_duplicates(subset=['Repository', 'Dataset', '% of Dataset removal', 'FS/HD', 'Hard_T', 'Training_Set%'], keep='last') \
                                .sort_values(by=['Repository', 'Dataset', 'Training Length', 'FS/HD'], ascending=[True, True, False, True]) \
                                .reset_index(drop=True)
    else:
        combined_data_accuracy = results_df_All_Short.reset_index(drop=True)

    combined_data_accuracy.to_excel(file_path_excel_accuracy, index=False)

    return results_df_All_Short



def save_results_All(path, df, is_fs, D_Counter):
    """
    Save the results to an Excel file.

    Args:
        path (str): The base path for saving the results.
        df (pd.DataFrame): The DataFrame containing the results.
        is_fs (str): Indicator whether the results are for feature selection ('FS') or not.
        D_Counter (float): The counter for dataset removal percentage.
    """
    # Calculate the percentage of dataset removal
    data_removal = round(D_Counter / 10, 2)
    df = df.rename(columns={'K': 'k'})
    if is_fs == 'FS':
        df['FS/HD'] = 'FS'
        df['% of Dataset removal'] = data_removal
       

    # List of algorithm columns to add
    algorithm_columns = [
        "ETree", "AdaBoost","SHAP", "Univariate", "low_variance", "SVC","SVM_RBF", "LS", "Skewness", "MCFS", "DT", "ReliefF", "alpha_investing",
        "DecisionTree_Forward", "DLFS", "SVM_Forward", "GRAPH", "RFS", "LeadingEV", "CAE", "mRMR", "CT", "GRACES", "UFS",
        "DecisionTree_Backward", "SVM_Backward"
    ]

    # Add algorithm columns to the DataFrame
    for algorithm in algorithm_columns:
        if is_fs == 'HD' and algorithm in ["ETree", "AdaBoost"]:
            # Extract the value from results_df_All_RF_loop
            algorithm_value = df.loc[df['Algorithm'] == algorithm, 'AUC ± Std_ovr'].values
            if len(algorithm_value) > 0:
                df[algorithm] = algorithm_value[0]
            else:
                df[algorithm] = None
        elif is_fs != 'HD' and algorithm not in ["ETree", "AdaBoost"]:
            # Extract the value from results_df_All_RF_loop
            algorithm_value = df.loc[df['Algorithm'] == algorithm, 'AUC ± Std_ovr'].values
            if len(algorithm_value) > 0:
                df[algorithm] = algorithm_value[0]
            else:
                df[algorithm] = None
        else:
            df[algorithm] = None

    # Define the file path for the Excel file
    file_path_excel = os.path.join(path, 'results', 'Convert Data 2 Hard', 'Change Log', 'Full_Results.xlsx')
    # Normalize the path and replace backslashes with forward slashes
    file_path_excel = os.path.normpath(file_path_excel).replace('\\', '/')

    # Save the combined data to the Excel file
    if os.path.exists(file_path_excel):
        existing_data = pd.read_excel(file_path_excel)
        combined_data = (
            pd.concat([existing_data, df])
            .drop_duplicates(
                subset=[
                    'Repository', 'Dataset', 'Hard_T', 'Training_Set%', 'N_Source', '# Features',
                    '% of Dataset removal', 'FS/HD', 'Leading_Algorithms_AUC', 'k'
                ],
                keep='last'
            )
            .sort_values(
                by=[
                    'Repository', 'Dataset', 'Hard_T', 'Training_Set%', 'N_Source', '# Features',
                    '% of Dataset removal', 'FS/HD', 'Leading_Algorithms_AUC', 'k'
                ],
                ascending=[True, True, False, False, True, True, True, True, True, True]
            )
            .reset_index(drop=True)
        )
    else:
        combined_data = df.reset_index(drop=True)

    # Save the combined data to the Excel file
    combined_data.to_excel(file_path_excel, index=False)


    # Optionally, save the combined data to a Parquet file
    # file_path_pq = os.path.join(path, 'Main/Output/Convert Data 2 Hard/Change Log/Results.parquet')
    # combined_data.to_parquet(file_path_pq, index=False)


def remove_phase_percent(train_data, train_labels, D_Counter):
    """
    Remove a percentage of samples from each class in the training data.

    Args:
        train_data (np.ndarray): The training data.
        train_labels (np.ndarray): The training labels.
        D_Counter (float): The counter for dataset removal percentage.

    Returns:
        np.ndarray: The updated training data.
        np.ndarray: The updated training labels.
        float: The updated D_Counter.
    """
    X = train_data
    Y = train_labels

    # Adjust labels if the minimum value is not 0
    #if np.min(Y) != 0 or not np.any(Y < 0):
    #    Y = Y - 1

    # Calculate the number of samples to remove per label
    unique_labels, counts = np.unique(Y, return_counts=True)
    to_remove_per_label = (counts * (D_Counter / 10)).astype(int)

    # Ensure at least one sample is removed
    if to_remove_per_label.sum() == 0:
        to_remove_per_label[0] = 1

    # Identify indices to remove
    indices_to_remove = []
    for label, to_remove in zip(unique_labels, to_remove_per_label):
        indices = np.where(Y == label)[0]
        np.random.shuffle(indices)
        indices_to_remove.extend(indices[:to_remove])

    # Remove the identified indices
    X_new = np.delete(X, indices_to_remove, axis=0)
    Y_new = np.delete(Y, indices_to_remove)

    # Print the original and new shapes of the data
    print("Original shapes:", X.shape, Y.shape)
    print("New shapes:", X_new.shape, Y_new.shape)

    return X_new, Y_new, D_Counter

def get_top_features(df, column_name='Best_indices_Sorted_AUC_ovo[Dataset/K&Hyperparamters]', top_n=60, fallback_n=10):
    """
    Extract the top features from a DataFrame column.

    Args:
        df (pd.DataFrame): The DataFrame containing the features.
        column_name (str): The name of the column containing the feature importance scores.
        top_n (int): The number of top features to extract if available.
        fallback_n (int): The number of features to extract if fewer than top_n features are available.

    Returns:
        pd.Series: The extracted top features.
    """
    try:
        # Check the number of features available
        num_features = len(df[column_name].iloc[0])
        
        # Determine the number of features to extract
        if num_features >= top_n:
            num_to_extract = top_n
        else:
            num_to_extract = fallback_n
        # Extract the top features
        top_features = df[column_name].iloc[:num_to_extract]

        return top_features

    except KeyError:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return pd.Series()

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.Series()



def calculate_feature_robustness(fs_features, top_features):
    """
    Calculate the robustness of feature selection by comparing two sets of features.

    Parameters:
    fs_features (list): List of features selected by a feature selection method.
    top_features (list): List of top features based on some criteria.

    Returns:
    tuple: A tuple containing common features, unique features, robustness indicator, and mutual information score.

    Methodology:
    What Can Be Considered as High Mutual Information:
    High Mutual Information: Generally, mutual information values are not bounded, but for normalized mutual information (NMI), values closer to 1 indicate a high degree of dependency. 
    In the context of feature selection, a high mutual information score (e.g., above 0.5) suggests that the features share a significant amount of information.
    Context-Dependent: The threshold for high mutual information can vary depending on the specific dataset and problem. It is often useful to compare mutual information scores relative to other features in the dataset.
    """
    from sklearn.feature_selection import mutual_info_classif

    try:
        # Flatten the numpy.ndarray and convert to a list if necessary
        if isinstance(top_features, np.ndarray):
            top_features = top_features.flatten().tolist()

        # Convert the list to a set
        top_features_set = set(top_features)

        # Compare the two sets of features
        fs_features_set = set(fs_features)

        # Find common features
        common_features = fs_features_set.intersection(top_features_set)

        # Find features unique to each set
        unique_to_fs = fs_features_set.difference(top_features_set)
        unique_to_top = top_features_set.difference(fs_features_set)

        # Calculate robustness indicator
        robustness_indicator = len(common_features) / len(fs_features_set)

        # Calculate robustness indicator using mutual information
        fs_features_list = list(fs_features_set)
        top_features_list = list(top_features_set)

        # Create a binary vector for each feature list
        all_features = list(set(fs_features_list + top_features_list))
        fs_binary_vector = [1 if feature in fs_features_list else 0 for feature in all_features]
        top_binary_vector = [1 if feature in top_features_list else 0 for feature in all_features]

        # Convert to numpy arrays
        fs_binary_vector = np.array(fs_binary_vector).reshape(-1, 1)
        top_binary_vector = np.array(top_binary_vector)

        # Calculate mutual information
        mutual_info = mutual_info_classif(fs_binary_vector, top_binary_vector, discrete_features=True)

        # Print the results
        print(f"Common features: {common_features}")
        print(f"Features unique to FS: {unique_to_fs}")
        print(f"Features unique to top features: {unique_to_top}")
        print(f"Robustness indicator Overlap: {robustness_indicator:.2f}")
        print(f"Robustness indicator Mutual Information: {mutual_info[0]:.2f}")

        return common_features, unique_to_fs, unique_to_top, robustness_indicator, mutual_info[0]

    except Exception as e:
        print(f"An error occurred: {e}")
        return set(), set(), set(), 0.0, 0.0


# Function to create new columns based on the specified column
def create_new_columns(df, column_name):
    df[f'{column_name}_Category'] = df[column_name].apply(lambda x: 'High' if x > 0.9 else 'Low' if x < 0.7 else 'Fair')
    return df



def get_dataset_labels(core_number):
    """
    Ensure the folder and Excel file exist, and read dataset labels from the Excel file.

    Args:
        convert_data_hard_path (str): The path for the 'Convert Data 2 Hard' folder.

    Returns:
        pd.DataFrame: The DataFrame containing the dataset labels.
    """
    #dataset_labels_path = os.path.join(convert_data_hard_path, 'Dataset Labels.xlsx')

    # Check if the folder exists, if not, create it
    if not os.path.exists(convert_data_hard_path_Source):
        os.makedirs(convert_data_hard_path_Source)
     
    convert_data_hard_path = convert_data_hard_path_Source +core_number+ 'Dataset Labels.xlsx'

    # Check if the Excel file exists, if not, create it with the specified columns
    if not os.path.exists(convert_data_hard_path):
        columns = [
                "Dataset", "# Instances Source(N)", "# Features(P)", "# Classes", "GT Dataset Label","Peeling Hard_T", "Initial Training Set %", "Experiment Label", "Experiment Comment", "Experiment Comment AUC", "Experiment Comment Accuracy",
                "GT_Training_Size", "GT_Test_Size",  "Initial Training Instances", "Test Instances",
                "Downsampling % HD Hard", "Training Instances HD Hard", "Downsampling % FS Hard", "Training Instances FS Hard", "AUC HD",
                "AUC With FS", "HD AUC Hard", "FS AUC Hard", "FS AUC with GT", "GT Contribution AUC", "GT Contribution % AUC",
                "Accuracy HD","Accuracy With FS", 
                 
                  "HD Accuracy Hard",
                "FS Accuracy Hard", "FS Accuracy with GT", "GT Contribution Accuracy", "GT Contribution Accuracy %", "FS Hard Indices AUC", "GT Indices AUC", "Common Features AUC",
                "Robustness Indicator Overlap AUC(Hard/GT)", "Robustness Indicator Mutual Information AUC (Hard/FS)",
                "FS Hard Indices Accuracy", "GT Indices Accuracy", "Common Features Accuracy",
                "Robustness Indicator Overlap Accuracy(Hard/GT)", "Robustness Indicator Mutual Information Accuracy (Hard/FS)",
                "Leading Algorithm HD", "Leading Algorithm Accuracy HD", "Leading Algorithm AUC FS", "Leading Algorithm Accuracy FS", "Leading Algorithm AUC Hard HD", "Leading Algorithm Accuracy Hard HD",
                "Leading Algorithm AUC Hard FS", "Leading Algorithm Accuracy Hard FS"
            ]
        df_empty = pd.DataFrame(columns=columns)
        df_empty.to_excel(convert_data_hard_path, index=False)

    # Read dataset labels from Excel file
    df_labels = pd.read_excel(convert_data_hard_path)

    return df_labels


def check_if_run_experiment(df_labels,  dataset, LowT, Training_size, rows, columns_len):
    """
    Check if the experiment has already been run.

    Args:
        df_labels (pd.DataFrame): The DataFrame containing the dataset labels.
        df_FS (pd.DataFrame): The DataFrame containing the feature selection results.
        dataset (str): The name of the dataset.
        LowT (float): The low threshold value.
        Training_size (float): The training set size percentage.
        rows (int): The number of rows in the dataset.
        columns_len (int): The number of columns in the dataset.

    Returns:
        bool: True if the experiment has already been run, False otherwise.
    """
    experiment_exists = df_labels[
        (df_labels['Dataset'] == dataset) &
        (df_labels['Peeling Hard_T'] == LowT) &
        (df_labels['Initial Training Set %'] == Training_size) &
        (df_labels['# Instances Source(N)'] == rows) &
        (df_labels['# Features(P)'] == columns_len)
    ]

    return not experiment_exists.empty




def update_dataset_labels_not_Hard(convert_data_hard_path, data_name, rows, K_Features, LowT, Training_size, Experiment_Label, Experiment_Comment, num_unique_labels, GT_Dataset_Label, HD_DownSampling, skip_to_next_dataset) :

    """
    Update the dataset labels DataFrame with new values.

    Args:
        data_name (str): The name of the dataset.
        HD_DownSampling (float): The HD downsampling percentage.
        D_Counter (float): The counter for dataset removal percentage.
        train_labels_new (np.ndarray): The new training labels.
        test_labels (np.ndarray): The test labels.
        LowT (float): The low threshold value.
        Training_size (float): The training set size percentage.
        FS_AlgoName (str): The name of the FS algorithm.
        convert_data_hard_path (str): The path to the dataset labels Excel file.
        skip_to_next_dataset (bool): Flag to indicate if the dataset should be skipped.
        rows (int): The number of rows in the dataset.
        num_unique_labels (int): The number of unique labels.
        Experiment_Label (str): The experiment label.
        Experiment_Comment (str): The experiment comment.
        FS_Dataset_Label (str): The FS dataset label.

    Returns:
        pd.DataFrame: The updated DataFrame.
    """
    # Read dataset labels from Excel file
    df_labels = pd.read_excel(convert_data_hard_path)


    # Define the columns in the desired order
    desired_columns =  [
                "Dataset", "# Instances Source(N)", "# Features(P)", "# Classes", "GT Dataset Label","Peeling Hard_T", "Initial Training Set %", "Experiment Label", "Experiment Comment", "Experiment Comment AUC", "Experiment Comment Accuracy",
                "GT_Training_Size", "GT_Test_Size",  "Initial Training Instances", "Test Instances",
                "Downsampling % HD Hard", "Training Instances HD Hard", "Downsampling % FS Hard", "Training Instances FS Hard", "AUC HD",
                "AUC With FS", "HD AUC Hard", "FS AUC Hard", "FS AUC with GT", "GT Contribution AUC", "GT Contribution % AUC",
                "Accuracy HD","Accuracy With FS", 
                 
                  "HD Accuracy Hard",
                "FS Accuracy Hard", "FS Accuracy with GT", "GT Contribution Accuracy", "GT Contribution Accuracy %", "FS Hard Indices AUC", "GT Indices AUC", "Common Features AUC",
                "Robustness Indicator Overlap AUC(Hard/GT)", "Robustness Indicator Mutual Information AUC (Hard/FS)",
                "FS Hard Indices Accuracy", "GT Indices Accuracy", "Common Features Accuracy",
                "Robustness Indicator Overlap Accuracy(Hard/GT)", "Robustness Indicator Mutual Information Accuracy (Hard/FS)",
                "Leading Algorithm HD", "Leading Algorithm Accuracy HD", "Leading Algorithm AUC FS", "Leading Algorithm Accuracy FS", "Leading Algorithm AUC Hard HD", "Leading Algorithm Accuracy Hard HD",
                "Leading Algorithm AUC Hard FS", "Leading Algorithm Accuracy Hard FS"
            ]



    # Create the DataFrame with the correct order
    new_row = pd.DataFrame([{
        "Dataset": data_name,
        "# Instances Source(N)": rows,
        "# Features(P)": K_Features,
        "# Classes": num_unique_labels,
        "GT Dataset Label": GT_Dataset_Label, 
        "Peeling Hard_T": LowT, 
        "Initial Training Set %": Training_size,  
        "Experiment Label": Experiment_Label,
        "Experiment Comment": Experiment_Comment,
        "Experiment Comment AUC": "",  # Blank
        "Experiment Comment Accuracy": "",  # Blank
        "GT_Training_Size": "",  # Blank
        "GT_Test_Size": "",  # Blank
        "Initial Training Instances": "",  # Blank
        "Test Instances": "",  # Blank
        "Downsampling % HD Hard": "",  # Blank
        "Training Instances HD Hard": "",  # Blank
        "Downsampling % FS Hard": "",  # Blank
        "Training Instances FS Hard": "",  # Blank
        "AUC HD": "",  # Blank
        "AUC With FS": "",  # Blank
        "HD AUC Hard": "",  # Blank
        "FS AUC Hard": "",  # Blank
        "FS AUC with GT": "",  # Blank
        "GT Contribution AUC": "",  # Blank
        "GT Contribution % AUC": "",  # Blank
        "Accuracy HD" : "",  # Blank
        "Accuracy With FS": "",  # Blank
        "HD Accuracy Hard": "",  # Blank
        "FS Accuracy Hard": "",  # Blank
        "FS Accuracy with GT": "",  # Blank
        "GT Contribution Accuracy": "",  # Blank
        "GT Contribution Accuracy %": "",  # Blank
        "FS Hard Indices AUC": "",  # Blank
        "GT Indices AUC": "",  # Blank
        "Common Features AUC": "",  # Blank
        "Robustness Indicator Overlap AUC(Hard/GT)": "",  # Blank
        "Robustness Indicator Mutual Information AUC (Hard/FS)": "",  # Blank
        "FS Hard Indices Accuracy": "",  # Blank
        "GT Indices Accuracy": "",  # Blank
        "Common Features Accuracy": "",  # Blank
        "Robustness Indicator Overlap Accuracy(Hard/GT)": "",  # Blank
        "Robustness Indicator Mutual Information Accuracy (Hard/FS)": "",  # Blank
        "Leading Algorithm HD": "",  # Blank
        "Leading Algorithm Accuracy HD": "",  # Blank
        "Leading Algorithm AUC FS": "",  # Blank
        "Leading Algorithm Accuracy FS": "",  # Blank
        "Leading Algorithm AUC Hard HD": "",  # Blank
        "Leading Algorithm Accuracy Hard HD": "",  # Blank
        "Leading Algorithm AUC Hard FS": "",  # Blank
        "Leading Algorithm Accuracy Hard FS": ""  # Blank
    }], columns=desired_columns)
    # Concatenate the new row with the existing DataFrame
    df_labels = pd.concat([df_labels, new_row], ignore_index=True)

    # Drop duplicates based on the specified columns
    df_labels = df_labels.drop_duplicates(subset=[
        "Dataset", "# Instances Source(N)", "# Features(P)", "# Classes", "GT Dataset Label", "Experiment Label", "Peeling Hard_T", "Initial Training Set %"
    ])

    # Save the updated DataFrame back to the Excel file
    df_labels.to_excel(convert_data_hard_path, index=False)

    return df_labels



def call_get_GT_Results(path_output_excel_FS):
    """
    Get the rows with the maximum Average AUC and Average Accuracy for each dataset and FS/HD.

    Args:
    - path_output_excel_FS (str): The path to the directory containing the Excel files.

    Returns:
    - pd.DataFrame: The combined rows with the maximum Average AUC and Average Accuracy for each dataset and FS/HD.
    """
    # Define the file path
    file_path = path_output_excel_FS + 'feature_selection_benchmark.xlsx'

    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name='Raw data', engine='openpyxl')

    # Filter the DataFrame to include only rows where 'FS/HD' is 'FS'
    df = df[df['FS/HD'] == 'FS']

    # Select the required columns

    columns_to_keep = [
        'K', 'FS/HD', 'Repository', 'Dataset', 'Majority_class_percentage', '#Classes', 'Class_Fraction', 'P', 'N', 'P/N',
        'P/N Classification', 'Algorithm', 'Hyperparamter', 'Train_size%', 'Average Accuracy', 'Std_dev_Accuracy', 'Acc ± Std',
        'Best_indices_Sorted_Acc[Dataset/K&Hyperparamters]', 'Average AUC', 'Std_dev_auc_ovr', 'AUC ± Std_ovr',
        'Best_indices_Sorted_AUC_ovr[Dataset/K&Hyperparamters]'
    ]

    # Suppose your DataFrame is called df
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]


    # Group by 'Dataset' and 'FS/HD' and get the rows with the maximum 'Average AUC'
    max_auc_ovr_rows = df.loc[df.groupby(['Dataset', 'FS/HD'])['Average AUC'].idxmax()]

    # Group by 'Dataset' and 'FS/HD' and get the rows with the maximum 'Average Accuracy'
    max_accuracy_rows = df.loc[df.groupby(['Dataset', 'FS/HD'])['Average Accuracy'].idxmax()]

    # Combine the rows with max Average AUC and max Average Accuracy
    combined_rows = pd.concat([max_auc_ovr_rows, max_accuracy_rows]).drop_duplicates()

    # Save the combined rows to a new Excel file
    light_results_file_path = path_output_excel_FS + 'Light_Results.xlsx'
    combined_rows.to_excel(light_results_file_path, index=False)

    print(f"Light results saved to {light_results_file_path}")

    return combined_rows



def calculate_feature_robustness(fs_features, top_features):
    """
    Calculate the robustness of feature selection by comparing two sets of features.
    
    Parameters:
    fs_features (list): List of features selected by a feature selection method.
    top_features (list): List of top features based on some criteria.
    
    Returns:
    tuple: A tuple containing common features, unique features, robustness indicator, and mutual information score.

    What Can Be Considered as High Mutual Information:
    High Mutual Information: Generally, mutual information values are not bounded, but for normalized mutual information (NMI), values closer to 1 indicate a high degree of dependency. In the context of feature selection, a high mutual information score (e.g., above 0.5) suggests that the features share a significant amount of information.
    Context-Dependent: The threshold for high mutual information can vary depending on the specific dataset and problem. It is often useful to compare mutual information scores relative to other features in the dataset.

    """
    from sklearn.feature_selection import mutual_info_classif

    # Flatten the numpy.ndarray and convert to a list if necessary
    if isinstance(top_features, np.ndarray):
        top_features = top_features.flatten().tolist()

    # Convert the list to a set
    top_features_set = set(top_features)

    # Compare the two sets of features
    fs_features_set = set(fs_features)

    # Find common features
    common_features = fs_features_set.intersection(top_features_set)

    # Find features unique to each set
    unique_to_fs = fs_features_set.difference(top_features_set)
    unique_to_top = top_features_set.difference(fs_features_set)

    # Calculate robustness indicator
    robustness_indicator = len(common_features) / len(fs_features_set)

    # Calculate robustness indicator using mutual information
    fs_features_list = list(fs_features_set)
    top_features_list = list(top_features_set)

    # Create a binary vector for each feature list
    all_features = list(set(fs_features_list + top_features_list))
    fs_binary_vector = [1 if feature in fs_features_list else 0 for feature in all_features]
    top_binary_vector = [1 if feature in top_features_list else 0 for feature in all_features]

    # Convert to numpy arrays
    fs_binary_vector = np.array(fs_binary_vector).reshape(-1, 1)
    top_binary_vector = np.array(top_binary_vector)

    # Calculate mutual information
    mutual_info = mutual_info_classif(fs_binary_vector, top_binary_vector, discrete_features=True)

    # Print the results
    print(f"Common features: {common_features}")
    print(f"Features unique to FS: {unique_to_fs}")
    print(f"Features unique to top features: {unique_to_top}")
    print(f" Robustness indicator Overlap: {robustness_indicator:.2f}")
    print(f"Robustness indicator Mutual Information: {mutual_info[0]:.2f}")

    return common_features, unique_to_fs, unique_to_top, robustness_indicator, mutual_info[0]

def Hard_Dataset_Documentation(
    train_data_source, train_labels_source,
    train_data_peeling, train_labels_peeling, test_data, test_labels, max_auc_ovr_rows_peeling_FS,
    max_auc_ovr_rows_FS_GT, GT_Dataset_Label, GT_Training_Size, GT_Test_Size, AUC_HD, AUC_FS, Acc_HD, Acc_FS,
    LowT, Training_size, Initial_Training_Instances, HD_DownSampling, Training_Instances_HD_Peeling, FS_DownSampling, 
    num_unique_labels, rows, K_Features, K_List, seed, hard_datasets_folder,
    HD_AUC_Hard,HD_Acc_Hard, FS_AUC_Hard, FS_Accuracy_Hard, convert_data_hard_path,
    algorithms_auc_HD, algorithms_accuracy_HD, algorithms_auc_FS, algorithms_accuracy_FS, algorithms_auc_peeling_FS, algorithms_accuracy_peeling_FS, algorithms_auc_peeling_HD, algorithms_accuracy_peeling_HD
):
    """
    Document the hard dataset by saving relevant data and calculating feature robustness.

    Args:
    - train_data_source (np.ndarray): Source training data.
    - train_labels_source (np.ndarray): Source training labels.
    - train_data_peeling (np.ndarray): Training data after peeling.
    - train_labels_peeling (np.ndarray): Training labels after peeling.
    - test_data (np.ndarray): Test data.
    - test_labels (np.ndarray): Test labels.
    - max_auc_ovr_rows_peeling_FS (pd.DataFrame): DataFrame containing max AUC OVR rows for peeling FS.
    - max_auc_ovr_rows_FS_GT (pd.DataFrame): DataFrame containing max AUC OVR rows for FS Ground Truth.
    - GT_Dataset_Label (str): Ground Truth dataset label.
    - GT_Training_Size (int): Ground Truth training size.
    - GT_Test_Size (int): Ground Truth test size.
    - AUC_HD (float): AUC for HD.
    - AUC_FS (float): AUC for FS.
    - Acc_HD (float): Accuracy for HD.
    - Acc_FS (float): Accuracy for FS.
    - LowT (float): Low threshold value.
    - Training_size (float): Training set size percentage.
    - Initial_Training_Instances (int): Initial number of training instances.
    - HD_DownSampling (float): HD downsampling rate.
    - Training_Instances_HD_Peeling (int): Number of training instances after HD peeling.
    - FS_DownSampling (float): FS downsampling rate.
    - num_unique_labels (int): Number of unique labels.
    - rows (int): Number of rows in the dataset.
    - K_Features (int): Number of features.
    - K_List (list): List of K values.
    - seed (int): Random seed for reproducibility.
    - hard_datasets_folder (str): Path to the folder where hard datasets are stored.
    - HD_AUC_Hard (float): HD AUC hard value.
    - FS_AUC_Hard (float): FS AUC hard value.
    - HD_Acc_Hard (float): HD Accuracy hard value.
    - convert_data_hard_path (str): The path to the dataset labels Excel file.
    - algorithms_auc_HD (list): List of AUC values for HD algorithms.
    - algorithms_accuracy_HD (list): List of accuracy values for HD algorithms.
    - algorithms_auc_FS (list): List of AUC values for FS algorithms.
    - algorithms_accuracy_FS (list): List of accuracy values for FS algorithms.
    - algorithms_auc_peeling_FS (list): List of AUC values for peeling FS algorithms.
    - algorithms_accuracy_peeling_FS (list): List of accuracy values for peeling FS algorithms.
    - algorithms_auc_peeling_HD (list): List of AUC values for peeling HD algorithms.
    - algorithms_accuracy_peeling_HD (list): List of accuracy values for peeling HD algorithms.

    Returns:
    - pd.DataFrame: The updated DataFrame containing experiment details and feature robustness metrics.
    """
    # Instances post FS Peeling 
    Training_Instances_FS_Peeling = len(train_labels_peeling)
    # Instances post Initial Test Set 
    Initial_Test_Instances = rows - Initial_Training_Instances

    # Section for FS Ground Truth (GT)
    FS_AUC_with_GT, FS_Accuracy_with_GT, indices_for_selected_GT_AUC, indices_for_selected_GT_Acc = process_fs_ground_truth(
        max_auc_ovr_rows_FS_GT, train_data_peeling, train_labels_peeling, test_data, test_labels, seed, num_unique_labels
    )

    # Finish Section for FS Ground Truth (GT)
    
    GT_Absolute_Contribution_AUC = (FS_AUC_with_GT - FS_AUC_Hard)
    # Calculate GT Improvement
    GT_Improvment_AUC = round((FS_AUC_with_GT / FS_AUC_Hard - 1) * 100, 3)

    GT_Absolute_Contribution_Acc = (FS_Accuracy_with_GT - FS_Accuracy_Hard)
    # Calculate GT Improvement
    GT_Improvment_Acc = round((FS_Accuracy_with_GT / FS_Accuracy_Hard - 1) * 100, 3)

    if GT_Improvment_AUC >= 10:
        Experiment_Label = "Hard Interesting"
        Experiment_Comment = "AUC with GT improved by 10% + AUC"
        Experiment_Comment_AUC = f"GT Contribution % AUC : {GT_Improvment_AUC}% (Hard->{round(FS_AUC_Hard * 100, 3)}% -> {round(FS_AUC_with_GT * 100, 3)}%)"
        Experiment_Comment_Accuracy = f"GT Contribution % Accuracy : {GT_Improvment_Acc}% (Hard :{round(FS_Accuracy_Hard * 100, 3)}% -> {round(FS_Accuracy_with_GT * 100, 3)}%)"
    else:
        Experiment_Label = "Hard Not Interesting"
        Experiment_Comment = "AUC with GT improvement is less than 10% + AUC"
        Experiment_Comment_AUC = f"GT Contribution % AUC : {GT_Improvment_AUC}% (Hard->{round(FS_AUC_Hard * 100, 3)}% -> {round(FS_AUC_with_GT * 100, 3)}%)"
        Experiment_Comment_Accuracy = f"GT Contribution % Accuracy : {GT_Improvment_Acc}% (Hard : {round(FS_Accuracy_Hard * 100, 3)}% -> {round(FS_Accuracy_with_GT * 100, 3)}%)"

    # Reset index and select only the first row
    max_auc_ovr_rows_peeling_FS = max_auc_ovr_rows_peeling_FS.reset_index(drop=True)

    # Extract values
    repository = max_auc_ovr_rows_peeling_FS['Repository'][0]
    dataset = max_auc_ovr_rows_peeling_FS['Dataset'][0]
    n = max_auc_ovr_rows_peeling_FS['N'][0]
    hard_t = LowT
    training_set_percent = Training_size
    n_source = rows
    num_features = K_Features
    k = K_List

    # Create the concatenated string
    concatenated_string = f"{repository}_{dataset}_{n}_{hard_t}_{training_set_percent}_{n_source}_{num_features}_{k}"

    # Print the concatenated string to verify
    print(concatenated_string)

    # Define the experiment folder path using Experiment_Label and concatenated_string
    experiment_folder = os.path.join(hard_datasets_folder, Experiment_Label, concatenated_string)

    # Check if the experiment folder exists, if not, create it
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder, exist_ok=True)
        print(f"Created folder: {experiment_folder}")

    # Save train_data_peeling and train_labels_peeling as a MATLAB file
    matlab_file_path_train = os.path.join(experiment_folder, "D_Train_Peeling.mat")
    scipy.io.savemat(matlab_file_path_train, {'X': train_data_peeling, 'Y': train_labels_peeling})
    print(f"Saved MATLAB file: {matlab_file_path_train}")

    # Save test_data and test_labels as a MATLAB file
    matlab_file_path_test = os.path.join(experiment_folder, "D_Test.mat")
    scipy.io.savemat(matlab_file_path_test, {'X': test_data, 'Y': test_labels})
    print(f"Saved MATLAB file: {matlab_file_path_test}")

    # Save train_data_peeling and train_labels_peeling as a MATLAB file
    matlab_file_path_train_source = os.path.join(experiment_folder, "D_Train_source_Pre_Peeling.mat")
    scipy.io.savemat(matlab_file_path_train_source, {'X': train_data_source, 'Y': train_labels_source}) 
    print(f"Saved MATLAB file: {matlab_file_path_train_source}")

    # Extract TOP Indices from peeling dataset by FS
    max_auc_ovr_rows_peeling_FS_for_indices = max_auc_ovr_rows_peeling_FS.reset_index(drop=True).iloc[0:1]
    FS_selected_features_FS_AUC = max_auc_ovr_rows_peeling_FS_for_indices['Best_indices_Sorted_AUC_ovr[Dataset/K&Hyperparamters]']
    # Extract the list from FS_selected_features_FS and convert it to a NumPy array
    FS_selected_features_FS_list_AUC = FS_selected_features_FS_AUC.iloc[0]
    FS_selected_features_FS_array_AUC = np.array(FS_selected_features_FS_list_AUC, dtype=int)

    FS_selected_features_FS_Accuracy = max_auc_ovr_rows_peeling_FS_for_indices['Best_indices_Sorted_Acc[Dataset/K&Hyperparamters]']
    # Extract the list from FS_selected_features_FS and convert it to a NumPy array
    FS_selected_features_FS_list_Accuracy = FS_selected_features_FS_Accuracy.iloc[0]
    FS_selected_features_FS_array_Accuracy = np.array(FS_selected_features_FS_list_Accuracy, dtype=int)

    # Convert indices_for_selected_GT to a NumPy array
    indices_for_selected_GT_AUC = np.array(indices_for_selected_GT_AUC, dtype=int)
    indices_for_selected_GT_Accuracy = np.array(indices_for_selected_GT_Acc, dtype=int)

    common_features_AUC, unique_to_fs, unique_to_top, robustness_indicator_Overlap_AUC, robustness_indicator_mutual_info_AUC = calculate_feature_robustness(FS_selected_features_FS_array_AUC, indices_for_selected_GT_AUC)
    common_features_Acc, unique_to_fs, unique_to_top, robustness_indicator_Overlap_Acc, robustness_indicator_mutual_info_Acc = calculate_feature_robustness(FS_selected_features_FS_array_Accuracy, indices_for_selected_GT_Accuracy)

    # Convert common_features_AUC to a list
    common_features_AUC_list = list(common_features_AUC)
    common_features_Acc_list = list(common_features_Acc)
    # Read dataset labels from Excel file
    df_labels = pd.read_excel(convert_data_hard_path)

    # Add a new row with the provided values   
    new_row = pd.DataFrame([{
        "Dataset": dataset,
        "# Instances Source(N)": rows,
        "# Features(P)": K_Features,
        "# Classes": num_unique_labels,
        "GT Dataset Label": GT_Dataset_Label,
        "Peeling Hard_T": LowT,
        "Initial Training Set %": Training_size,
        "Experiment Label": Experiment_Label,
        "Experiment Comment": Experiment_Comment,
        "Experiment Comment AUC": Experiment_Comment_AUC,
        "Experiment Comment Accuracy": Experiment_Comment_Accuracy,
        "GT_Training_Size": GT_Training_Size,
        "GT_Test_Size": GT_Test_Size,
        "Initial Training Instances": Initial_Training_Instances,
        "Test Instances": Initial_Test_Instances,
        "Downsampling % HD Hard": HD_DownSampling,
        "Training Instances HD Hard": Training_Instances_HD_Peeling,
        "Downsampling % FS Hard": FS_DownSampling,
        "Training Instances FS Hard": Training_Instances_FS_Peeling,
        "AUC HD": AUC_HD,
        "AUC With FS": AUC_FS,
        "HD AUC Hard": HD_AUC_Hard,
        "FS AUC Hard": FS_AUC_Hard,
        "FS AUC with GT": FS_AUC_with_GT,
        "GT Contribution AUC": GT_Absolute_Contribution_AUC,
        "GT Contribution % AUC": f"{GT_Improvment_AUC}%",  # Format as percentage
        "Accuracy HD": Acc_HD,
        "Accuracy With FS": Acc_FS,
        "HD Accuracy Hard": HD_Acc_Hard,
        "FS Accuracy Hard": FS_Accuracy_Hard,
        "FS Accuracy with GT": FS_Accuracy_with_GT,
        "GT Contribution Accuracy": GT_Absolute_Contribution_Acc,
        "GT Contribution Accuracy %": f"{GT_Improvment_Acc}%",  # Format as percentage
        "FS Hard Indices AUC": FS_selected_features_FS_array_AUC.tolist(),  # Convert array to list
        "GT Indices AUC": indices_for_selected_GT_AUC.tolist(),  # Convert array to list
        "Common Features AUC": common_features_AUC_list,  # Convert array to list
        "Robustness Indicator Overlap AUC(Hard/GT)": robustness_indicator_Overlap_AUC,
        "Robustness Indicator Mutual Information AUC (Hard/FS)": robustness_indicator_mutual_info_AUC,
        "FS Hard Indices Accuracy": FS_selected_features_FS_array_Accuracy.tolist(),  # Convert array to list
        "GT Indices Accuracy": indices_for_selected_GT_Accuracy.tolist(),  # Convert array to list
        "Common Features Accuracy": common_features_Acc_list,  # Convert array to list
        "Robustness Indicator Overlap Accuracy(Hard/GT)": robustness_indicator_Overlap_Acc,
        "Robustness Indicator Mutual Information Accuracy (Hard/FS)": robustness_indicator_mutual_info_Acc,
        "Leading Algorithm HD": algorithms_auc_HD,
        "Leading Algorithm Accuracy HD": algorithms_accuracy_HD,
        "Leading Algorithm AUC FS": algorithms_auc_FS,
        "Leading Algorithm Accuracy FS": algorithms_accuracy_FS,
        "Leading Algorithm AUC Hard HD": algorithms_auc_peeling_HD,
        "Leading Algorithm Accuracy Hard HD": algorithms_accuracy_peeling_HD,
        "Leading Algorithm AUC Hard FS": algorithms_auc_peeling_FS,
        "Leading Algorithm Accuracy Hard FS": algorithms_accuracy_peeling_FS
    }])

    # Concatenate the new row with the existing DataFrame
    df_labels = pd.concat([df_labels, new_row], ignore_index=True)

    # Save the updated DataFrame back to the Excel file
    df_labels.to_excel(convert_data_hard_path, index=False)

    return df_labels

def process_fs_ground_truth(max_auc_ovr_rows_FS_GT, train_data_peeling, train_labels_peeling, test_data, test_labels, seed, num_unique_labels):
    """
    Process FS Ground Truth (GT) section.

    Args:
    - max_auc_ovr_rows_FS_GT (pd.DataFrame): DataFrame containing max AUC OVR rows for FS Ground Truth.
    - train_data_peeling (np.ndarray): Training data after peeling.
    - train_labels_peeling (np.ndarray): Training labels after peeling.
    - test_data (np.ndarray): Test data.
    - test_labels (np.ndarray): Test labels.
    - seed (int): Random seed for reproducibility.
    - num_unique_labels (int): Number of unique labels.

    Returns:
    - float: FS AUC with GT.
    - float: FS Accuracy with GT.
    - np.ndarray: Indices for selected GT AUC.
    - np.ndarray: Indices for selected GT Accuracy.
    """
    # Reset index and select only the first row
    max_auc_ovr_rows_FS_GT = max_auc_ovr_rows_FS_GT.reset_index(drop=True).iloc[0:1]

    # Extract the desired columns
    GT_selected_features_AUC = max_auc_ovr_rows_FS_GT['Best_indices_Sorted_AUC_ovr[Dataset/K&Hyperparamters]']
    GT_selected_features_Accuracy = max_auc_ovr_rows_FS_GT['Best_indices_Sorted_Acc[Dataset/K&Hyperparamters]']

    # Process AUC
    GT_selected_features_list_AUC = ast.literal_eval(GT_selected_features_AUC.iloc[0])
    GT_selected_features_array_AUC = np.array(GT_selected_features_list_AUC, dtype=int)
    indices_for_selected_GT_AUC, feature_importance, accumulated_importance = calculate_feature_importance_rf(train_data_peeling, train_labels_peeling, GT_selected_features_array_AUC, seed)
    train_data_peeling_selected_AUC = np.take(train_data_peeling, indices_for_selected_GT_AUC, axis=1)
    test_data_selected_AUC = np.take(test_data, indices_for_selected_GT_AUC, axis=1)
    clf = ExtraTreesClassifier(n_estimators=200, max_depth=3, random_state=seed)
    clf.fit(train_data_peeling_selected_AUC, train_labels_peeling)
    if num_unique_labels == 2:
        test_predicted_probs = clf.predict_proba(test_data_selected_AUC)[:, 1]
        Ground_Truth_test_auc_ovo = round(roc_auc_score(test_labels, test_predicted_probs), 3)
        Ground_Truth_test_auc_ovr = Ground_Truth_test_auc_ovo.copy()
    else:
        test_predicted_probs = clf.predict_proba(test_data_selected_AUC)
        Ground_Truth_test_auc_ovo = round(roc_auc_score(test_labels, test_predicted_probs, multi_class='ovo'), 3)
        Ground_Truth_test_auc_ovr = round(roc_auc_score(test_labels, test_predicted_probs, multi_class='ovr'), 3)
    FS_AUC_with_GT = Ground_Truth_test_auc_ovr

    # Process Accuracy
    GT_selected_features_list_Acc = ast.literal_eval(GT_selected_features_Accuracy.iloc[0])
    GT_selected_features_array_Acc = np.array(GT_selected_features_list_Acc, dtype=int)
    indices_for_selected_GT_Acc, feature_importance, accumulated_importance = calculate_feature_importance_rf(train_data_peeling, train_labels_peeling, GT_selected_features_array_Acc, seed)
    train_data_peeling_selected_Acc = np.take(train_data_peeling, indices_for_selected_GT_Acc, axis=1)
    test_data_selected_Acc = np.take(test_data, indices_for_selected_GT_Acc, axis=1)
    clf.fit(train_data_peeling_selected_Acc, train_labels_peeling)
    accuracy = round(accuracy_score(test_labels, clf.predict(test_data_selected_Acc)), 3)
    FS_Accuracy_with_GT = accuracy

    return FS_AUC_with_GT, FS_Accuracy_with_GT, indices_for_selected_GT_AUC, indices_for_selected_GT_Acc