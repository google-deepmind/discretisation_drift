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
"""Computing inception score and FID using TF-GAN."""
from absl import logging

import jax
import jax.numpy as jnp
import numpy as np
import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub
import tqdm

from dd_two_player_games import metric_utils as metrics


_one_key_split = lambda key: tuple(jax.random.split(key))


def split_key(key, no_pmap):
  if no_pmap:
    return _one_key_split(key)
  else:
    return jax.pmap(_one_key_split)(key)


def assert_data_ranges(data, eps=1e-3):
  # Check data is in [-1, 1]
  assert jnp.min(data) >= -1 - eps, jnp.min(data)
  assert jnp.max(data) <= 1 + eps, jnp.max(data)

  assert jnp.min(data) < -0.01, jnp.min(data)
  assert jnp.max(data) > 0.01, jnp.max(data)


DEFAULT_DTYPES = {tfgan.eval.INCEPTION_OUTPUT: tf.float32,
                  tfgan.eval.INCEPTION_FINAL_POOL: tf.float32}


def classifier_fn_from_tfhub(
    inception_model, output_fields, return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Wrapping the TF-Hub module in another function defers loading the module until
  use, which is useful for mocking and not computing heavy default arguments.

  Args:
    inception_model: A model loaded from TFHub.
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


class InceptionMetrics(object):
  """Object which holds onto the Inception net.

  Data and samples need to be in [-1, 1].
  """

  def __init__(self, dataset=None, num_splits=10,
               check_data_ranges=False):

    self.dataset = dataset
    self.num_splits = num_splits
    self.is_initialized = False  # Delay some startup away from the constructor.
    self.check_data_ranges = check_data_ranges
    self.inception_model = None

    logging.info(
        'Splitting inception data/samples in %d splits', num_splits)

  def initialize(self):
    """Load weights from CNS and pmap/construct the forward pass."""
    if not self.is_initialized:
      self.inception_model = tfhub.load(tfgan.eval.INCEPTION_TFHUB)

      @tf.function
      def run_inception(x):
        classifier_fn = classifier_fn_from_tfhub(
            self.inception_model,
            output_fields=[
                tfgan.eval.INCEPTION_OUTPUT,
                tfgan.eval.INCEPTION_FINAL_POOL])
        return tfgan.eval.run_classifier_fn(
            x, classifier_fn, dtypes=DEFAULT_DTYPES)

      def run_net(x):
        size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
        if x.ndim == 5:
          # If we have an extra dimension because of pmap merge the
          # batch and device dimension.
          shape = x.shape
          x = np.reshape(x, (-1, shape[2], shape[3], shape[4]))
        x = tf.image.resize(x, [size, size])
        inception_output = run_inception(x)
        logits = inception_output['logits']
        pool = inception_output['pool_3']
        logits = tf.nn.log_softmax(logits)
        return logits, pool

      self.run_net = run_net

      self.data_mus, self.data_sigmas = None, None
      self.is_initialized = True

  def get_metrics(
      self, sampler, num_inception_images, batch_size, rng, no_pmap=True,
      sample_postprocess_fn=None):
    """Calculate inception score and FID on a set of samples.

    Args:
      sampler: A sample_utils.Sampler object which can be generate samples.
      num_inception_images: The number of samples to produce per metric.
        In total, the number of samples used is `num_inception_images`
        x `num_splits` in order to create standard deviations around the
        metrics. A reasonable number is 10K. Note that for FID, the number
        of samples is important since FID is a biased metric: the more
        samples you have, the better the metric value. This does not entail
        that the model performs better.
      batch_size: The batch size to use on each iteration of the inception net,
        taken as the overall batch size (rather than per-device).
      rng: An rng to use with a haiku network.
      no_pmap: Whether functions should operate assuming a num_devices axis.
      sample_postprocess_fn: The forward pass of the generator might produce
        more than just a single batch of images. This function, if none, is
        applied to the output of the generator before being passed to Inception.
    Returns:
      A dict with the keys 'IS' and 'FID'
    """
    self.initialize()

    num_samples = num_inception_images * self.num_splits
    logging.info(
        'Number of samples used to compute each metric value: %d',
        num_inception_images)
    logging.info('Total number of samples used: %d', num_samples)
    num_batches = int(np.ceil(num_samples / batch_size))

    if self.data_mus is None:
      self.data_mus, self.data_sigmas, _, _ = self.get_dataset_statistics(
          self.dataset, num_batches=num_batches)

    # Loop over all batches and get pool + logits
    logits, pool = [], []
    for _ in range(num_batches):
      rng, step_rng = (split_key(rng, no_pmap) if rng is not None
                       else (None, None))
      samples = sampler.sample_batch(batch_size, rng=step_rng)
      if sample_postprocess_fn:
        samples = sample_postprocess_fn(samples)

      if self.check_data_ranges:
        assert_data_ranges(samples)

      this_logits, this_pool = self.run_net(samples)  # pytype: disable=attribute-error

      pool += [this_pool]
      logits += [this_logits]

    logits = np.concatenate(logits, 0)[:num_samples]
    pool = np.concatenate(pool, 0)[:num_samples]

    logging.info('Calculating FID...')
    fid_mean, fid_std = metrics.calculate_frechet_distance_with_error_bars(
        pool, self.data_mus, self.data_sigmas)
    logging.info('Calculating Inception scores...')
    is_mean, is_std = metrics.calculate_inception_scores_with_error_bars(
        logits, self.num_splits, use_jax=False)
    return {'IS_mean': is_mean, 'IS_std': is_std,
            'FID': np.array(fid_mean), 'FID_std': np.array(fid_std)}

  def get_dataset_statistics(self, dataset, num_batches):
    """Loop over all samples in a dataset and calc pool means and covs.

    Args:
      dataset: an iterator which returns appropriately batched images and
        labels, with the images in [-1, 1] and shaped (device x NHWC)
      num_batches: the number of batches to use for the computation of the
        statistics.
    Returns:
      The mean and covariance stats computed across the entire dataset.
    """
    self.initialize()
    batch = 0
    logits, pool = [], []
    for x in tqdm.tqdm(dataset):
      if self.check_data_ranges:
        assert_data_ranges(x)

      if batch == num_batches:
        break
      this_logits, this_pool = self.run_net(x)
      pool += [this_pool]
      logits += [this_logits]
      batch += 1

    logging.info('Finished iterating, concatenating pool and logits')
    pool = np.concatenate(pool, 0)
    logits = np.concatenate(logits, 0)
    logging.info('Pool shape %s, logits shape %s', pool.shape, logits.shape)
    logging.info('Calculating Inception scores...')
    IS_mean, IS_std = metrics.calculate_inception_scores_with_error_bars(  # pylint: disable=invalid-name
        logits, self.num_splits, use_jax=False)
    logging.info('Inception score on dataset is %f +/- %f.', IS_mean, IS_std)

    logging.info('Calculating means and sigmas on dataset...')

    mus = []
    sigmas = []
    chunk_size = (pool.shape[0] // self.num_splits)
    for index in range(self.num_splits):
      mus.append(np.mean(
          pool[index * chunk_size: (index + 1) * chunk_size], axis=0))
      sigmas.append(np.cov(
          pool[index * chunk_size: (index + 1) * chunk_size], rowvar=False))

    return mus, sigmas, pool, logits


