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
"""Datasets for GAn training."""


import collections

import tensorflow as tf
import tensorflow_datasets as tfds


DataProcessor = collections.namedtuple(
    'DataProcessor', ['preprocess', 'postprocess'])


def image_preprocess(image):
  """Convert to floats in [0, 1]."""
  image = tf.cast(image['image'], tf.float32) / 255.0
  # Scale the data to [-1, 1] to stabilize training.
  return 2.0 * image - 1.0


def image_postprocess(image):
  return (image + 1.) / 2


ImageProcessor = DataProcessor(image_preprocess, image_postprocess)


def get_image_dataset(
    dataset_name, split, batch_dims, processor, shard_args, seed=1):
  """Read and shuffle dataset.

  Args:
    dataset_name: the name of the dataset as expected by Tensorflow datasets.
    split: `train`, `valid` or `test.
    batch_dims: the batch dimensions to be used for the dataset. Can be
      an integer or a list of integers. A list of integers is often used if
      parallel training is used.
    processor: a `DataProcessor` instance.
    shard_args: Argument to be passed to dataset to TensorFlow datasets
      `shard`. A tuple of size 2, the first one is the number of shards
      (often corresponding to the number of devices), and the second one
      is the index in the number of shards that is used by the current device.
      In jax, often the right value is [jax.host_count(), jax.host_id()].
    seed: the random seed passed to `ds.shuffle`.

  Returns:
    An iterator of numpy arrays, with the leading dimensions given by
    `batch_dims`.
  """
  ds = tfds.load(dataset_name, split=split)

  ds = ds.shard(*shard_args)

  # Shuffle before repeat ensures all examples seen in an epoch.
  # See https://www.tensorflow.org/guide/data_performance#repeat_and_shuffle.
  ds = ds.shuffle(buffer_size=10000, seed=seed)
  ds = ds.repeat()

  ds = ds.map(processor.preprocess,
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

  for batch_size in batch_dims[::-1]:
    ds = ds.batch(batch_size, drop_remainder=True)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  return ds.as_numpy_iterator()
