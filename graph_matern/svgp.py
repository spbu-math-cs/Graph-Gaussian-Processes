# Copyright (c) 2020 Viacheslav Borovitskiy, Iskander Azangulov, Alexander Terenin, Peter Mostowsky, Marc Peter Deisenroth, Nicolas Durrande. All Rights Reserved.

from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.svgp import SVGP
import tensorflow as tf


class GraphSVGP(SVGP):
    """SVGP via Graph Fourier feature approximations. See section 3.1 in https://arxiv.org/pdf/2010.15538.pdf.
        GraphSVGP makes VI for coefficients of eigenvectors of Graph Laplacian"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = self.eval_MV_from_q(q_mu, q_sqrt, Xnew, full_cov)
        return mu + self.mean_function(Xnew), var

    def eval_MV_from_q(self, q_mu, q_sqrt, X, full_cov=False):
        """Build the posterior mean and variance from q_mu, q_sqrt"""

        X_id = tf.reshape(tf.cast(X[:, 0], dtype=tf.int32), [-1, 1])
        S = self.kernel.eval_S(self.kernel.kappa, self.kernel.sigma_f)
        U = tf.gather_nd(self.kernel.eigenvectors, X_id) * S[None, :]
        mu = tf.einsum('ij,jl->il', U, q_mu)
        if q_sqrt.shape.ndims == 3:
            q_cov = tf.einsum('ijn,kjn->ijn', q_sqrt, q_sqrt)
        if q_sqrt.shape.ndims == 2:
            q_cov = tf.einsum('in,in->in', q_sqrt, q_sqrt)
        if full_cov:
            if q_sqrt.shape.ndims == 3:
                var = tf.einsum('ij,njk,lk->nil', U, q_cov, U)
            if q_sqrt.shape.ndims == 2:
                var = tf.einsum('ij, jn, kj->nik', U, q_cov, U)
        else:
            if q_sqrt.shape.ndims == 3:
                var = tf.einsum('ij,njk,ik->in', U, q_cov, U)
            if q_sqrt.shape.ndims == 2:
                var = tf.einsum('ij, jn,ij->in', U, q_cov, U)
        return mu, var
