#!/bin/sh
# Copyright 2021 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e
python3 -m venv dd_venv
echo 'Created venv'
source dd_venv/bin/activate

pip install --upgrade pip



echo 'Getting requirements.'
pip install -r requirements.txt

# Install haiku from source (update once)
pip install git+https://github.com/deepmind/dm-haiku

# Change the cuda version to the one you want to use.
pip install --upgrade "jax[cpu]==0.2.20"
# pip install --upgrade "jax[cuda112]==0.2.20" -f https://storage.googleapis.com/jax-releases/jax_releases.html


if [[ $1 == "train" ]]; then
  echo 'Starting training...'
  PYTHONPATH=.::$PYTHONPATH python dd_two_player_games/experiment.py \
    --config=dd_two_player_games/sgd_cifar_config.py --logtostderr;
elif [[ $1 == "eval" ]]; then
  echo 'Starting evaluation...'
  PYTHONPATH=.::$PYTHONPATH python dd_two_player_games/experiment.py \
    --config=dd_two_player_games/sgd_cifar_config.py --logtostderr \
    --jaxline_mode eval;
else
  echo 'Invalid option: pass in argument train or eval to this script!'
fi
