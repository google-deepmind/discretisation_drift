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

"""Model utilities."""
import collections
import immutabledict

import jax


class Sampler(object):
  """Sampler used for obtaining evaluation metric."""

  def __init__(self, model, params, state, pmap=True):
    self.model = model
    self.params = params
    self.state = state
    self.pmap = pmap

  def sample_batch(self, batch_size, rng):
    gen_kwargs = immutabledict.immutabledict(
        collections.OrderedDict([('is_training', False),
                                 ('test_local_stats', False)]))
    sample_fn = self.model.sample
    if self.pmap:
      sample_fn = jax.pmap(
          sample_fn, axis_name='i', static_broadcasted_argnums=(3, 4))
    return sample_fn(
        self.params.gen, self.state.players.gen, rng, batch_size, gen_kwargs)[0]
