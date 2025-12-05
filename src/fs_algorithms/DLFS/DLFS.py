import os
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

import sys
# Add the directory containing FeatureSelector.py to the Python path
import sys, os


#from .FeatureSelector import FeatureSelector
from FeatureSelector import FeatureSelector

import numpy as np
import random as rn
import os



def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

def DLFS_FS(data_name,  p_train_feature, p_train_label, p_test_feature, p_test_label, key_feture_number, row, random_state,device):
    
    """
    Perform FeatureImportanceDL feature selection based on NN.
    This function selects the top 'k' features based on the importances from the mask.

    Args:
        data_name (str): Name of the dataset.
        p_train_feature (numpy array): Training data with features.
        p_train_label (numpy array): Labels corresponding to the training data.
        p_test_feature (numpy array): Test data with features.
        p_test_label (numpy array): Labels corresponding to the test data.
        key_feature_number (int): Maximum number of top features to select.
        row (dict): Dictionary containing hyperparameters.
        random_state (int): Seed used by the random number generator for reproducibility.

    Returns:
        list: Indices of the top 'k' important features.
    """
    import tensorflow as tf

    # Set seed for reproducibility
    seed = random_state
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)


    try:
        # First attempt with test_size=0.2
        x_train, x_validate, y_train, y_validate = train_test_split(
            p_train_feature, p_train_label, stratify=p_train_label, test_size=0.2, random_state=seed
        )
    except ValueError as e:
        print(f"Error with test_size=0.2: {e}")
        try:
            # Second attempt with test_size=0.5
            x_train, x_validate, y_train, y_validate = train_test_split(
                p_train_feature, p_train_label, stratify=p_train_label, test_size=0.5, random_state=seed
            )
        except ValueError as e:
            print(f"Error with test_size=0.5: {e}")
            try:
                print("Duplicate single-member classes to ensure stratification.")
                # Convert p_train_label to a list if it's not already
                if isinstance(p_train_label, np.ndarray):
                    p_train_label = p_train_label.tolist()

                # Duplicate single-member classes
                unique, counts = np.unique(p_train_label, return_counts=True)
                single_member_classes = unique[counts == 1]

                for cls in single_member_classes:
                    indices = np.where(np.array(p_train_label) == cls)[0]
                    p_train_label.extend([p_train_label[i] for i in indices])
                    p_train_feature = np.vstack([p_train_feature, p_train_feature[indices]])

                # Now perform the train-test split again
                x_train, x_validate, y_train, y_validate = train_test_split(
                    p_train_feature, p_train_label, stratify=p_train_label, test_size=0.5, random_state=seed
                )

            except ValueError as e:
                if "The least populated class in y has only 1 member" in str(e):
                    print("Error: The least populated class in y has only 1 member, which is too few.")
                    indices_for_selected = ['no indices']
                    return indices_for_selected
                else:
                    raise e

    x_test = p_test_feature
    y_test = p_test_label

    y_tr = get_one_hot(np.array(y_train).astype(np.int8), len(np.unique(p_train_label)))
    y_te = get_one_hot(np.array(y_test).astype(np.int8), len(np.unique(p_train_label)))
    y_val = get_one_hot(np.array(y_validate).astype(np.int8), len(np.unique(p_train_label)))

    # Pick a PyTorch device (CUDA on PCs, MPS on Apple, else CPU)
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    elif torch.cuda.is_available():
        torch_device = torch.device("cuda:0")
    else:
        torch_device = torch.device("cpu")

    # Convert the PyTorch device to a TensorFlow device string, **by asking TF**
    import tensorflow as tf

    def tf_pick_device_from_torch(torch_dev: torch.device) -> str:
        # If Torch says "gpu-ish" (cuda or mps), try to use TF GPU if present.
        if torch_dev.type in {"cuda", "mps"}:
            gpus = tf.config.list_logical_devices("GPU")
            if gpus:
                # e.g. '/device:GPU:0' on Apple Metal and on CUDA
                return gpus[0].name
        # Fallback
        return "/device:CPU:0"

    tf_device = tf_pick_device_from_torch(torch_device)

    print("Torch device:", torch_device)
    print("TF device:", tf_device)



    with tf.device(tf_device):

        fs = FeatureSelector((p_train_feature.shape[1],), row['s'], row['data_batch_size'], row['mask_batch_size'], str_id=data_name)
        fs.create_dense_operator([60, 30, 20, len(np.unique(p_train_label))], row['activation'], metrics=[keras.metrics.CategoricalAccuracy()])
        fs.operator.set_early_stopping_params(row['phase_2_start'], patience_batches=row['early_stopping_patience'], minimize=True)
        fs.create_dense_selector([100, 50, 10, 1])
        fs.create_mask_optimizer(epoch_condition=row['phase_2_start'], perturbation_size=row['s_p'])
        fs.train_networks_on_data(x_train, y_tr, row['max_batches'], val_data=(x_validate, y_val))

    importances, optimal_mask = fs.get_importances(return_chosen_features=True)
    optimal_subset = np.nonzero(optimal_mask)
    indices_for_selected = np.argpartition(importances, -key_feture_number)[-key_feture_number:].tolist()


    
    return indices_for_selected

    """

    k = key_feture_number
    # Normalize importance scores based on all features
    normalized_importances = importances / np.sum(np.abs(importances))
    rounded_importances = np.round(normalized_importances, 3)

    # Select top 'k' features based on the normalized importance scores
    k = min(len(importances), k)  # Ensure k does not exceed the number of features
    selected_features = np.argsort(-rounded_importances)[:k]

    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.

    # Calculate accumulated importance of the selected features
    accumulated_importance = np.round(np.sum(rounded_importances[top_k_indices]), 3)
    if accumulated_importance > 1:  # Correcting potential rounding error causing the sum to exceed 1
        accumulated_importance = 1.0

    return list(top_k_indices), list(rounded_importances[top_k_indices]), accumulated_importance

    """
