import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K


# ---------- helper layer --------------------------------------------------
class ConcreteSelect(Layer):
    def __init__(self, k, num_features,
                 start_temp=10.0, min_temp=0.1, alpha=0.99999, **kw):
        super().__init__(**kw)
        self.k            = k
        self.num_features = num_features
        self.start_temp   = float(start_temp)
        self.min_temp     = tf.constant(min_temp, dtype=tf.float32)
        self.alpha        = tf.constant(alpha,  dtype=tf.float32)
        self.expand_out   = Dense(num_features)      # keeps output shape

    # ---------------------------------------------------------------------
    def build(self, input_shape):
        # temperature scalar (not-trainable)
        self.temp = self.add_weight(
            name='temp',                 # <-- keyword!
            shape=(),                    # scalar
            initializer=Constant(self.start_temp),
            trainable=False)

        # logits matrix (trainable)
        self.logits = self.add_weight(
            name='logits',               # <-- keyword!
            shape=(self.k, self.num_features),
            initializer='glorot_uniform',
            trainable=True)

        super().build(input_shape)


    # ---------------------------------------------------------------------
    def call(self, inputs, training=None):
        # --- Gumbel-softmax sample ---------------------------------------
        gumbel  = -tf.math.log(-tf.math.log(
                     tf.random.uniform(tf.shape(self.logits),
                                       K.epsilon(), 1.0)))
        curr_T  = tf.maximum(self.min_temp, self.temp * self.alpha)
        softsel = tf.nn.softmax((self.logits + gumbel) / curr_T, axis=-1)
        hardsel = tf.one_hot(tf.argmax(softsel, axis=1),
                             depth=self.num_features)

        # -------- robust switch for eager / graph mode -------------------
        if training is None:                          # happens during tracing
            training = K.learning_phase()            # 0 = infer, 1 = train
        sel = tf.where(tf.cast(training, tf.bool),   # bool scalar
                       softsel, hardsel)

        z = tf.matmul(inputs, sel, transpose_b=True)     # (batch, k)
        return self.expand_out(z)                        # (batch, p)

    # <- NEW: tell Keras the shape so it stops complaining
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_features)


    # ---------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(k=self.k, num_features=self.num_features,
                        start_temp=self.start_temp,
                        min_temp=float(self.min_temp.numpy()),
                        alpha=float(self.alpha.numpy())))
        return cfg


# ---------- early stopping on temperature “cool-down” ---------------------
class StopperCallback(EarlyStopping):
    def __init__(self, thresh=0.998):
        super().__init__(monitor='__dummy__', patience=np.iinfo(int).max,
                         verbose=0, mode='max', baseline=thresh)

    def get_monitor_value(self, logs=None):
        logits = self.model.get_layer('concrete_select').logits
        m = K.mean(K.max(K.softmax(logits, axis=-1), axis=-1))
        return K.get_value(m)


# -------------------------------------------------------------------------
# Concrete auto-encoder feature selector (dup-free, importance accessible)
# -------------------------------------------------------------------------
class ConcreteAutoencoderFeatureSelector:
    def __init__(self, K, output_function, num_features,
                 num_epochs=300, batch_size=None, learning_rate=1e-3,
                 start_temp=10.0, min_temp=0.1):
        self.K              = int(K)
        self.output_fn      = output_function
        self.num_features   = int(num_features)
        self.num_epochs     = int(num_epochs)
        self.batch_size     = batch_size
        self.lr             = learning_rate
        self.start_temp     = start_temp
        self.min_temp       = min_temp
        self.model          = None

    # ---------------  internal helpers  ----------------------------------
    @staticmethod
    def _to_2d(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {x.shape}")
        return x.astype("float32")

    def _chosen_indices(self):
        """
        Return the indices of the K strongest columns (unique).
        Works both before and after model training.
        """
        # get the (K, p) numpy matrix – NOT the scalar temp
        logits = K.get_value(self.model.get_layer("concrete_select").logits)
        scores = logits.mean(axis=0)                # length p
        return np.argsort(scores)[-self.K:][::-1]   # top-K descending


    # ---------------  public API  ----------------------------------------
    def fit(self, X, y, val_X=None, val_y=None):
        
        # Convert design matrix ------------------------------------------------
        X = self._to_2d(X).astype("float32")

        # Convert target -------------------------------------------------------
        y = np.asarray(y, dtype="float32")
        if y.ndim == 1:                     # label vector → (n,1)
            y = y.reshape(-1, 1)
        # otherwise keep the original 2-D shape (auto-encoder case)

        #X = self._to_2d(X);  y = np.asarray(y, dtype="float32").reshape(-1, 1)
        if self.batch_size is None:
            self.batch_size = max(len(X)//256, 16)

        steps  = (len(X)+self.batch_size-1)//self.batch_size
        alpha  = math.exp(math.log(self.min_temp/self.start_temp) /
                          (self.num_epochs*steps))

        selector = ConcreteSelect(self.K, self.num_features,
                                  self.start_temp, self.min_temp, alpha,
                                  name="concrete_select")

        inp  = Input(shape=(self.num_features,))
        out  = self.output_fn(selector(inp))
        self.model = Model(inp, out)
        self.model.compile(optimizer=Adam(self.lr), loss="mse")

        val = (self._to_2d(val_X), np.asarray(val_y)) if val_X is not None else None
        self.model.fit(X, y,
                       batch_size=self.batch_size,
                       epochs=self.num_epochs,
                       validation_data=val,
                       callbacks=[StopperCallback()],
                       verbose=0)
        return self

    def transform(self, X):
        X = self._to_2d(X)
        return X[:, self._chosen_indices()]

    def fit_transform(self, X, y, **kw):
        self.fit(X, y, **kw)
        return self.transform(X)

    # ---------------- convenience getters --------------------------------
    def get_support(self, indices=False):
        idx = self._chosen_indices()
        if indices:
            return idx
        mask = np.zeros(self.num_features, dtype=bool);  mask[idx] = True
        return mask

    def get_feature_importance(self):
        """
        Importance = column-wise absolute-weight sum of your named 'output_layer'.
        Length == K and index order matches _chosen_indices().
        """
        idx = self._chosen_indices()

        # weight matrix of the output Dense layer: shape (hidden_units, p)
        w   = self.model.get_layer("output_layer").get_weights()[0]

        # aggregate importance for the selected columns
        return np.abs(w[:, idx]).sum(axis=0)

    # ---------------------------------------------------------------------
    def _chosen_indices(self):
        """
        Return the indices of the K strongest columns (unique).
        Works both before and after model training.
        """
        # get the (K, p) numpy matrix – NOT the scalar temp
        logits = K.get_value(self.model.get_layer("concrete_select").logits)
        scores = logits.mean(axis=0)                # length p
        return np.argsort(scores)[-self.K:][::-1]   # top-K descending
