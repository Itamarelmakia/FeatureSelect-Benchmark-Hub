import os
import numpy as np
import random as rn
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Layer
from keras.models import Model
from keras import optimizers, initializers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
#from . import Functions as F 
import Functions as F 
from sklearn.ensemble import RandomForestClassifier
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)


def UFS_FS(p_train_feature, p_train_label, p_test_feature, p_test_label, key_feture_number, row, random_state,device):
    """
    Perform feature selection using the UFS_FS algorithm.

    Args:
        X_train (numpy array): Training data with features.
        y_train (numpy array): Labels corresponding to the training data.
        p_test_feature (numpy array): Test data with features.
        p_test_label (numpy array): Labels corresponding to the test data.
        key_feture_number (int): Maximum number of top features to select.
        row (dict): Dictionary containing hyperparameters for the UFS_FS algorithm.
        random_state (int): Seed used by the random number generator for reproducibility.

    Returns:
        tuple: (C_Selected_X_Train, C_Selected_X_Test, y_train, sorted_indices, sorted_importance, accumulated_importance)
            - C_Selected_X_Train (numpy array): Compressed training data.
            - C_Selected_X_Test (numpy array): Compressed test data.
            - y_train (numpy array): Labels corresponding to the training data.
            - selected_features (list): Indices of the top 'k' important features.

    """
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
    class Feature_Select_Layer(Layer):
    
            def __init__(self, output_dim, **kwargs):
                super(Feature_Select_Layer, self).__init__(**kwargs)
                self.output_dim = output_dim
    
            def build(self, input_shape):
                self.kernel = self.add_weight(name='kernel',
                                              shape=(input_shape[1],),
                                              initializer=initializers.RandomUniform(minval=0.999999, maxval=0.9999999,
                                                                                     seed=seed),
                                              trainable=True)
                super(Feature_Select_Layer, self).build(input_shape)
    
            def call(self, x, selection=False, k=key_feture_number):
                kernel = tf.pow(self.kernel, 2)
                if selection:
                    kernel_ = tf.transpose(kernel)
                    kth_largest = tf.math.top_k(kernel_, k=k)[0][-1]
                    kernel = tf.where(condition=tf.less(kernel, kth_largest), x=tf.zeros_like(kernel), y=kernel)
                return tf.linalg.matmul(x, tf.linalg.diag(kernel))
    
    
            def compute_output_shape(self, input_shape):
                return (input_shape[0], self.output_dim)
    
        # --------------------------------------------------------------------------------------------------------------------------------
    def Autoencoder(p_data_feature=x_train.shape[1], \
                        p_encoding_dim=key_feture_number, \
                        p_learning_rate=1E-3):
            input_img = Input(shape=(p_data_feature,), name='input_img')
    
            encoded = Dense(p_encoding_dim, activation='linear', kernel_initializer=initializers.glorot_uniform(seed))(
                input_img)
            bottleneck = encoded
            decoded = Dense(p_data_feature, activation='linear', kernel_initializer=initializers.glorot_uniform(seed))(
                encoded)
    
            latent_encoder = Model(input_img, bottleneck)
            autoencoder = Model(input_img, decoded)
    
            autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=p_learning_rate))
    
          #  print('Autoencoder Structure-------------------------------------')
           # autoencoder.summary()
            return autoencoder, latent_encoder
    
        # --------------------------------------------------------------------------------------------------------------------------------
    def Identity_Autoencoder(p_data_feature=x_train.shape[1], \
                                 p_encoding_dim=key_feture_number, \
                                 p_learning_rate=1E-3):
            input_img = Input(shape=(p_data_feature,), name='autoencoder_input')
    
            feature_selection = Feature_Select_Layer(output_dim=p_data_feature, \
                                                     input_shape=(p_data_feature,), \
                                                     name='feature_selection')
    
            feature_selection_score = feature_selection(input_img)
    
            encoded = Dense(p_encoding_dim, \
                            activation='linear', \
                            kernel_initializer=initializers.glorot_uniform(seed), \
                            name='autoencoder_hidden_layer')
    
            encoded_score = encoded(feature_selection_score)
    
            bottleneck_score = encoded_score
    
            decoded = Dense(p_data_feature, \
                            activation='linear', \
                            kernel_initializer=initializers.glorot_uniform(seed), \
                            name='autoencoder_output')
    
            decoded_score = decoded(bottleneck_score)
    
            latent_encoder_score = Model(input_img, bottleneck_score)
            autoencoder = Model(input_img, decoded_score)
    
            autoencoder.compile(loss='mean_squared_error', \
                                optimizer=optimizers.Adam(learning_rate=p_learning_rate))
    
         #   print('Autoencoder Structure-------------------------------------')
            autoencoder.summary()
            return autoencoder, latent_encoder_score
    
        # --------------------------------------------------------------------------------------------------------------------------------
    def Fractal_Autoencoder(p_data_feature=x_train.shape[1], \
                                p_feture_number=key_feture_number, \
                                p_encoding_dim=key_feture_number, \
                                p_learning_rate=1E-3, \
                                p_loss_weight_1=1, \
                                p_loss_weight_2=2):
            input_img = Input(shape=(p_data_feature,), name='autoencoder_input')
    
            feature_selection = Feature_Select_Layer(output_dim=p_data_feature, \
                                                     input_shape=(p_data_feature,), \
                                                     name='feature_selection')
    
            feature_selection_score = feature_selection(input_img)
            feature_selection_choose = feature_selection(input_img, selection=True, k=p_feture_number)
    
            encoded = Dense(p_encoding_dim, \
                            activation='linear', \
                            kernel_initializer=initializers.glorot_uniform(seed), \
                            name='autoencoder_hidden_layer')
    
            encoded_score = encoded(feature_selection_score)
            encoded_choose = encoded(feature_selection_choose)
    
            bottleneck_score = encoded_score
            bottleneck_choose = encoded_choose
    
            decoded = Dense(p_data_feature, \
                            activation='linear', \
                            kernel_initializer=initializers.glorot_uniform(seed), \
                            name='autoencoder_output')
    
            decoded_score = decoded(bottleneck_score)
            decoded_choose = decoded(bottleneck_choose)
    
            latent_encoder_score = Model(input_img, bottleneck_score)
            latent_encoder_choose = Model(input_img, bottleneck_choose)
            feature_selection_output = Model(input_img, feature_selection_choose)
            autoencoder = Model(input_img, [decoded_score, decoded_choose])
    
            autoencoder.compile(loss=['mean_squared_error', 'mean_squared_error'], \
                                loss_weights=[p_loss_weight_1, p_loss_weight_2], \
                                optimizer=Adam(learning_rate=p_learning_rate))
    
        #    print('Autoencoder Structure-------------------------------------')
            #autoencoder.summary()
            return autoencoder, feature_selection_output, latent_encoder_score, latent_encoder_choose
    
    
        # Running The Model:
        
    epochs_number = row['epochs_number'] # All scores run with 200
    batch_size_value = row['batch_size_value'] # 16

    loss_weight_1 = row['loss_weight_1']  # 0.0078125
    F_AE, \
    feature_selection_output, \
    latent_encoder_score_F_AE, \
    latent_encoder_choose_F_AE = Fractal_Autoencoder(p_data_feature=x_train.shape[1], \
                                                        p_feture_number=key_feture_number, \
                                                        p_encoding_dim=key_feture_number, \
                                                        p_learning_rate=row['p_learning_rate'], \
                                                        p_loss_weight_1=loss_weight_1, \
                                                        p_loss_weight_2=1)

    F_AE_history = F_AE.fit(x_train, [x_train, x_train], \
                            epochs=epochs_number, \
                            batch_size=batch_size_value, \
                            shuffle=True, \
                            validation_data=(x_validate, [x_validate, x_validate]),verbose=0)

    p_data = F_AE.predict(x_test, verbose=0)
    numbers = x_test.shape[0] * x_test.shape[1]
    FS_layer_output = feature_selection_output.predict(x_test, verbose=0)
    key_features,top_k_idx = F.top_k_keepWeights_1(F_AE.get_layer(index=1).get_weights()[0], key_feture_number)

    p_seed = seed
    selected_position_list = np.where(key_features > 0)[0]
    C_Selected_X_Train = p_train_feature[:, selected_position_list]
    C_Selected_X_Test = p_test_feature[:, selected_position_list]

    return x_train, p_test_feature, y_train,selected_position_list


    """
    Comment: Another option is to use the feature importance from the AdaBoost classifier to select the top 'k' features. For standardization, we use the Etree for all algorithms.


    # Calculate feature importance with ExtraTreesClassifier after feature selection
    import sys
    sys.path.append(utilites_path)
    from utilities import calculate_feature_importance_rf

    # Calculate feature importance with RandomForest after feature selection
    sorted_indices, sorted_importance_rounded, accumulated_importance = calculate_feature_importance_rf(X_train, y_train, selected_features)

    return C_Selected_X_Train, C_Selected_X_Test, y_train, selected_features

    """