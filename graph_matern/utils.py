# Copyright (c) 2020 Viacheslav Borovitskiy, Iskander Azangulov, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth, Nicolas Durrande. All Rights Reserved.

""" Utility functions. """
import warnings
import tensorflow as tf


def tf_jitchol(mat, jitter=0, dtype=tf.float32):
    """Run Cholesky decomposition with an increasing jitter,
    until the jitter becomes too large.
    Arguments
    ---------
    mat : (m, m) tf.Tensor
        Positive-definite matrix
    jitter : float
        Initial jitter
    """
    try:
        chol = tf.linalg.cholesky(mat)
        return chol
    except:
        new_jitter = jitter*10.0 if jitter > 0.0 else 1e-15
        if new_jitter > 1.0:
            raise RuntimeError('Matrix not positive definite even with jitter')
        warnings.warn(
            'Matrix not positive-definite, adding jitter {:e}'
            .format(new_jitter),
            RuntimeWarning)
        new_jitter = tf.cast(new_jitter, dtype=dtype)
        return tf_jitchol(mat + tf.multiply(new_jitter, tf.eye(mat.shape[-1], dtype=dtype)), new_jitter)
