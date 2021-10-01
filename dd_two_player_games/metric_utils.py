# Copyright 2021 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to compute GAN evaluation metrics."""

import jax.numpy as jnp
import numpy as np
import scipy.linalg


def calculate_inception_scores_with_error_bars(
    pred, num_splits=10, use_jax=True):
  """Calculate the Inception Score using multiple splits.

  Args:
    pred: The logits produced when running InceptionV3 on a set of samples,
      not including the FC layer's bias (for some reason), then passing said
      logits through a log softmax.
    num_splits: Number of splits to compute mean and std of IS over.
    use_jax: Accelerate these calculations with jax?
  Returns:
    The mean and standard deviation of the Inception Score on each split.
  """
  onp = jnp if use_jax else np
  scores = []
  chunk_size = (pred.shape[0] // num_splits)
  for index in range(num_splits):
    pred_chunk = pred[index * chunk_size: (index + 1) * chunk_size]
    scores.append(calculate_inception_score(pred_chunk, use_jax=use_jax))
  return onp.mean(scores), onp.std(scores)


def calculate_inception_score(pred, use_jax=True):
  onp = jnp if use_jax else np
  log_mean = onp.log(onp.expand_dims(onp.mean(onp.exp(pred), 0), 0))
  kl_inception = onp.exp(pred) * (pred - log_mean)
  kl_inception = onp.mean(onp.sum(kl_inception, 1))
  return onp.exp(kl_inception)


def calculate_frechet_distance_with_error_bars(
    pool, data_mus, data_sigmas, use_jax=True):
  """Calculate the Frechet distance using multiple splits.

  Args:
    pool: The pool3 features produced when running InceptionV3 on a set of
      samples.
    data_mus: A list of `num_split` means obtained from the data,
      to be compared with those obtained from samples.
    data_sigmas: A list of `num_split` covariances obtained from the data,
      to be compared with those obtained from samples.
    use_jax: Accelerate these calculations with jax?
  Returns:
    The mean and standard deviation of the Inception Score on each split.
  """
  assert len(data_mus) == len(data_sigmas)
  num_splits = len(data_mus)

  onp = jnp if use_jax else np
  scores = []
  chunk_size = (pool.shape[0] // num_splits)
  for index in range(num_splits):
    pool_chunk = pool[index * chunk_size: (index + 1) * chunk_size]
    mean = onp.mean(pool_chunk, axis=0)
    cov = onp.cov(pool_chunk, rowvar=False)
    scores.append(calculate_frechet_distance(
        mean, cov, data_mus[index], data_sigmas[index]))

  return onp.mean(np.array(scores)), onp.std(np.array(scores))


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
  """JAX implementation of the Frechet Distance.

  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.

  Args:
    mu1: Sample mean over activations from distribution 1 (samples by default).
    sigma1: Covariance matrix over activations from dist 1 (samples by default).
    mu2: Sample mean over activations from distribution 2 (dataset by default).
    sigma2: Covariance matrix over activations from dist 2 (dataset by default).

  Returns:
    The Frechet Distance.
  """
  diff = mu1 - mu2
  # Run 25 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
  covmean = np.real(scipy.linalg.sqrtm(np.matmul(sigma1, sigma2)))
  out = (jnp.dot(diff, diff) +  jnp.trace(sigma1) + jnp.trace(sigma2)
         - 2 * jnp.trace(covmean))
  return out
