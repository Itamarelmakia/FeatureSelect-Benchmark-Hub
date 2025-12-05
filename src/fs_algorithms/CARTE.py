# pip install fasttext-wheel # CARTE-AI can be installed from PyPI:
# pip install fastparquet==2024.5.0

import numpy as np
import os
import random as rn
import torch
from sklearn.preprocessing import StandardScaler
from carte_table_to_graph import *
from carte_estimator import *




def CARTE_FS(X_train, y_train , X_test, y_test, K, row, random_state):


    """
    Perform feature selection using the CARTE methodology to determine the importance of features within a dataset.

    This function automatically utilizes GPU if available, otherwise falls back to CPU. It is especially designed
    for high-dimensional, sparse datasets susceptible to the curse of dimensionality.

    Args:
        X_train (numpy.ndarray): The training feature matrix.
        X_test (numpy.ndarray): The testing feature matrix.
        y_train (numpy.ndarray): The training target variable.
        y_test (numpy.ndarray): The testing target variable.
        K (int): The number of features to select.
        row (dict): Dictionary containing hyperparameters for the model and additional settings.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        list: Indices of the top 'K' important features.
    """
    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    
    # Check available CPU cores for parallel processing
    available_cores = os.cpu_count()  # Get the total number of available CPU cores
    n_jobs = min(row.get('n_jobs', available_cores), available_cores)  # Use the minimum of specified n_jobs and available cores
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    

    # Convert table data to graph format suitable for the CARTE model
    preprocessor = Table2GraphTransformer()
    X_train = preprocessor.fit_transform(X_train, y=y_train)
    X_test = preprocessor.transform(X_test)

    # Determine if GPU is available and not currently used
    if torch.cuda.is_available() and not torch.cuda.current_device():
        device = 'cuda'
    else:
        device = 'cpu'

    # Prepare the CARTE model configuration
    carte_params = {
        "num_model": row.get('num_model', 10),
        "disable_pbar": row.get('disable_pbar', False),
        "random_state": seed,
        "device": device,
        "n_jobs": n_jobs
    }
    
    # Initialize and train the CARTE model
    estimator = CARTEClassifier(**carte_params)
    estimator.fit(X=X_train, y=y_train)

    # Extract and sort feature importances
    feature_importances = estimator.feature_importances_
    top_k_indices = np.argsort(feature_importances)[::-1][:K]

    return top_k_indices.tolist()