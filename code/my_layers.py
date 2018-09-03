import keras.backend as K
from keras.engine.topology import Layer
from keras.layers.convolutional import Conv1D
from keras import initializers
from keras import regularizers
from keras import constraints
import tensorflow as tf
import numpy as np


################################################################################
# Quadratic-time MMD with Gaussian RBF

def _mix_rbf_kernel(X, Y, sigmas=[1.], wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(tf.shape(K_XX)[0], tf.float32)
    n = tf.cast(tf.shape(K_YY)[0], tf.float32)

    
    if biased:
        mmd2 = (tf.reduce_sum(K_XX, keep_dims=True) / (m * m)
              + tf.reduce_sum(K_YY, keep_dims=True) / (n * n)
              - 2 * tf.reduce_sum(K_XY, keep_dims=True) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2


def mix_rbf_mmd2(X, Y, sigmas=[1.], wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def rbf_mmd2(X, Y, sigma=1., biased=True):
    return mix_rbf_mmd2(X, Y, sigmas=[sigma], biased=biased)


################################################################################


################################################################################
# Customized layers

class Max_over_time(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Max_over_time, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            x = x * mask
        return K.max(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None

class KL_loss(Layer):
    def __init__(self, batch_size, **kwargs):
        super(KL_loss, self).__init__(**kwargs)
        self.batch_size = batch_size


    def call(self, x, mask=None):
        a = x[0]
        b = x[1]

        a = K.mean(a, axis=0, keepdims=True)
        b = K.mean(b, axis=0, keepdims=True)
        a /= K.sum(a, keepdims=True)
        b /= K.sum(b, keepdims=True)

        a = K.clip(a, K.epsilon(), 1)
        b = K.clip(b, K.epsilon(), 1)

        loss = K.sum(a*K.log(a/b), axis=-1, keepdims=True) \
            + K.sum(b*K.log(b/a), axis=-1, keepdims=True)

        loss = K.repeat_elements(loss, self.batch_size, axis=0)
        
        return loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def compute_mask(self, x, mask):
        return None


class mmd_loss(Layer):
    def __init__(self, batch_size, **kwargs):
        super(mmd_loss, self).__init__(**kwargs)
        self.batch_size = batch_size


    def call(self, x, mask=None):
        a = x[0]
        b = x[1]

        mmd = rbf_mmd2(a, b)
        mmd = K.repeat_elements(mmd, self.batch_size, axis=0)

        return mmd

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def compute_mask(self, x, mask):
        return None



class Ensemble_pred_loss(Layer):
    def __init__(self, **kwargs):
        super(Ensemble_pred_loss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        pred = x[0]
        target = x[1]
        weight = x[2]

        error = K.categorical_crossentropy(target, pred)
        loss = error * weight

        return loss

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def compute_mask(self, x, mask):
        return None



class Conv1DWithMasking(Conv1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DWithMasking, self).__init__(**kwargs)
    
    def compute_mask(self, x, mask):
        return mask


