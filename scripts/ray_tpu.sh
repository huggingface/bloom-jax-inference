# copied from https://github.com/kingoflolz/mesh-trzansformer-jax

#!/usr/bin/env bash
set -e

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c 'import time; time.sleep(999999999)'

# initializes jax and installs ray on cloud TPUs
sudo pip install "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip install --upgrade ray[default]==1.5.1 transformers fabric dataclasses optax flax tqdm cloudpickle smart_open[gcs] einops func_timeout