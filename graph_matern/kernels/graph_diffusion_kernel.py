# Copyright (c) 2020 Viacheslav Borovitskiy, Iskander Azangulov, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth, Nicolas Durrande. All Rights Reserved.

import numpy as np
import tensorflow as tf
import gpflow
from graph_matern.utils import tf_jitchol


class GraphDiffusionKernel(gpflow.kernels.Kernel):
    """Implementation of diffusion kernel on graph. Kernel is direct product of diffusion on graph and some kernel on \R^d
        Note that this kernel equivalent to Matern Kernel with \nu=\infty

    Attributes
    ----------

    eigenpairs : tuple
        Truncated tuple returned by tf.linalg.eigh applied to the Laplacian of the graph.
    kappa : float
        Trainable lengthscale hyperparameter.
    sigma_f : float
        Trainable scaling kernel hyperparameter.
    vertex_dim: int
        dimension of \R^d
    point_kernel: gpflow.kernels.Kernel
        kernel on \R^d, None if vertex_dim is 0
    active_dims: slice or list of indices, None by default
        gpflow.kernel.Kernel parameter.
    dtype : tf.dtypes.DType
        type of tensors, tf.float64 by default
        """

    def __init__(self, eigenpairs, kappa=4, sigma_f=1,
                 vertex_dim=0, point_kernel=None, active_dims=None, dtype=tf.float64):

        self.eigenvectors, self.eigenvalues = eigenpairs
        self.num_verticies = tf.cast(tf.shape(self.eigenvectors)[0], dtype=dtype)
        self.vertex_dim = vertex_dim
        if vertex_dim != 0:
            self.point_kernel = point_kernel
        else:
            self.point_kernel = None
        self.dtype = dtype

        self.kappa = gpflow.Parameter(kappa, dtype=self.dtype, transform=gpflow.utilities.positive(), name='kappa')
        self.sigma_f = gpflow.Parameter(sigma_f, dtype=self.dtype, transform=gpflow.utilities.positive(), name='sigma_f')
        super().__init__(active_dims=active_dims)

    def eval_S(self, kappa, sigma_f):
        S = tf.exp(-0.5*self.eigenvalues*kappa*kappa)
        S = tf.multiply(S, self.num_verticies/tf.reduce_sum(S))
        S = tf.multiply(S, sigma_f)
        return S

    def _eval_K_vertex(self, X_id, X2_id):
        if X2_id is None:
            X2_id = X_id

        S = self.eval_S(self.kappa, self.sigma_f)

        K_vertex = (tf.gather_nd(self.eigenvectors, X_id) * S[None, :]) @ \
            tf.transpose(tf.gather_nd(self.eigenvectors, X2_id))  # shape (n,n)

        return K_vertex

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        if self.vertex_dim == 0:
            X_id = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1])
            X2_id = tf.reshape(tf.cast(X2[:, 0], dtype=tf.int32), [-1, 1])
            K = self._eval_K_vertex(X_id, X2_id)

        else:
            X_id, X_v = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1]), X[:, 1:]
            X2_id, X2_v = tf.reshape(tf.cast(X2[:, 0], dtype=tf.int32), [-1, 1]), X2[:, 1:]

            K_vertex = self._eval_K_vertex(X_id, X2_id)
            K_point = self.point_kernel.K(X_v, X2_v)

            K = tf.multiply(K_point, K_vertex)

        return K

    def _eval_K_diag_vertex(self, X_id):
        S = self.eval_S(self.kappa, self.sigma_f)

        K_diag_vertex = tf.reduce_sum(tf.transpose((tf.gather_nd(self.eigenvectors, X_id)) * S[None, :]) *
                        tf.transpose(tf.gather_nd(self.eigenvectors, X_id)), axis=0)

        return K_diag_vertex

    def K_diag(self, X):
        if self.vertex_dim == 0:
            X_id = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1])
            K_diag = self._eval_K_diag_vertex(X_id)
        else:
            X_id, X_v = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1]), X[:, 1:]
            K_diag_vertex = self._eval_K_diag_vertex(X_id)
            K_diag_point = self.point_kernel.K_diag(X_v)
            K_diag = K_diag_point * K_diag_vertex
        return K_diag

    def sample(self, X):
        K_chol = tf_jitchol(self.K(X), dtype=self.dtype)
        sample = K_chol.dot(np.random.randn(tf.shape(K_chol)[0]))
        return sample
