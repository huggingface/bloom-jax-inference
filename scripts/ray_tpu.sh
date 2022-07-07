# copied from https://github.com/kingoflolz/mesh-trzansformer-jax

#!/usr/bin/env bash
set -e

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c 'import time; time.sleep(999999999)'

# TODO: this should go in a venv
# TODO: add checks to see if installation is already done, if yes, skip it.

# initializes jax and installs ray on cloud TPUs
# sudo pip install --user "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# sudo pip install --user --upgrade ray==1.13.0 transformers fabric dataclasses optax flax tqdm func_timeout

sudo pip install -e bloom_inference/

# TODO: this should be it's own command.
TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368 ray start --address=$1 --resources="{\"tpu\": 1}"