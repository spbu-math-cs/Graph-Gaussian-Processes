# Copyright (c) 2020 Viacheslav Borovitskiy, Iskander Azangulov, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth, Nicolas Durrande. All Rights Reserved.

import tensorflow as tf

import gpflow
from gpflow.inducing_variables import InducingVariables
from gpflow.base import TensorLike
from gpflow import covariances as cov
from gpflow.kernels import Kernel


class GPInducingVariables(InducingVariables):
    """
       Graph inducing points.
       The first coordinate is vertex index.
       Other coordinates are matched with points on \R^d.
       Note that vertex indices are not trainable.
    """
    def __init__(self, x, out_dim):
        self.x_id = x[:, :1]
        if len(x.shape) > 1:
            self.x_v = gpflow.Parameter(x[:, 1:], dtype=gpflow.default_float())

        self.shape_ = list(x.shape)
        print(self.shape_)
        self.shape_.append(out_dim)
        self.shape_ = tf.convert_to_tensor(self.shape_, dtype=tf.int32)

        self.N = self.x_id.shape[0]

    def __len__(self):

        return self.N

    @property
    def GP_IV(self):
        return tf.concat([self.x_id, self.x_v], axis=1)

    @property
    def num_inducing(self) -> tf.Tensor:
        return self.x_id.shape[0]

    @property
    def shape(self):
        return self.shape_

@cov.Kuu.register(GPInducingVariables, gpflow.kernels.Kernel)
def Kuu_kernel_GPinducingvariables(
        inducing_variable: InducingVariables,
        kernel: Kernel,
        jitter=0.0):
    GP_IV = inducing_variable.GP_IV

    Kuu = kernel.K(GP_IV)
    Kuu += jitter * tf.eye(tf.shape(Kuu)[0], dtype=Kuu.dtype)

    return Kuu


@cov.Kuf.register(GPInducingVariables, gpflow.kernels.Kernel, TensorLike)
def Kuf_kernel_GPinducingvariables(
        inducing_variable: InducingVariables,
        kernel: Kernel,
        X: tf.Tensor):
    GP_IV = inducing_variable.GP_IV

    Kuf = kernel.K(GP_IV, X)

    return Kuf

