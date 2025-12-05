import numpy as np
import random as rn

#from concrete_autoencoder import ConcreteAutoencoderFeatureSelector # As the auther used Keras with old version we make many updates to make it also for tenserflow with 2.1 version
#from .Autoencoders import ConcreteAutoencoderFeatureSelector

from Autoencoders import ConcreteAutoencoderFeatureSelector

from keras.layers import Dense, Dropout, LeakyReLU, Softmax
from keras.utils import to_categorical



import os
import sys

# Set up paths
path = os.getcwd()
# Add the 'Algorithms' directory to the system path
sys.path.append(os.path.join(path, 'Algorithms'))



def CAE_FS(x_train, x_test, key_feature_number, row, random_state, device):
    """
    Perform feature selection using the Concrete AutoEncoder (CAE) algorithm on a specified device.
    
    Args:
        x_train (np.ndarray): Training data.
        x_test (np.ndarray): Test data.
        key_feature_number (int): Number of features to select.
        row (dict): Dictionary containing hyperparameters.
        random_state (int): Seed for reproducibility.
        device (str): Device to run the model on (e.g. '/GPU:0', '/GPU:1', or '/CPU:0').
        
    Returns:
        list: A list of unique selected feature indices.
    """
    import os
    import numpy as np
    import random as rn
    import tensorflow as tf
    from tensorflow.keras import backend as K
    
    # Set seeds for reproducibility.
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    rn.seed(random_state)
    
    """
    This is the decoder of the developers but we found that in some complex dataset (i,e Zoo where number of feature are limited we get duplicated indices and the K value is not reached) 
    in addtion we want to add the feature importance vased on the weihgts of the output layer.
    def decoder(x):
        x = Dense(320)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(320)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(x_train.shape[1])(x)
        return x
    """
    def decoder(x):
        # First encoding layer with 320 units, LeakyReLU activation, and dropout
        x = Dense(320)(x)
        x = LeakyReLU(0.2)(x)  # Using LeakyReLU for better gradient flow in case of negative inputs
        x = Dropout(0.1)(x)  # Dropout to prevent overfitting by randomly setting input units to 0

        # Second encoding layer replicated as above for depth and complexity
        x = Dense(320)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)

        # Output layer: used for decoding, directly influences feature importance
        # Naming this layer allows for easy identification later for extracting weights
        output_layer = Dense(x_train.shape[1], activation='sigmoid', name='output_layer')(x)
        return output_layer



    def decoder_supervised(x):
        x = Dense(320)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(320)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(x_test.shape[1])(x)
        x = Softmax()(x)
        return x

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



        selector = ConcreteAutoencoderFeatureSelector(
                    K=key_feature_number,
                    output_function=decoder,        # your existing decoder
                    num_features=x_train.shape[1],
                    num_epochs=800)

        selector.fit(x_train, x_train)               # unsupervised AE
        chosen          = selector.get_support(indices=True)      # unique K indices
        feature_imp     = selector.get_feature_importance()       # same length K
        #print("Chosen columns:", chosen)
        #print("Importance:", feature_imp)

    return chosen
