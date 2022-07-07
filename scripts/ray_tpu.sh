#!/usr/bin/env bash
set -e

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c "import time; time.sleep(999999999)"

# testing... remove venv each time
rm -rf venv

# clear cache -> currently limited to 100 GB storage
rm -rf ~/.cache

# check if venv exists
if [ -d ~/venv]
then
  echo "venv exists"
  # activate venv (if not done so already)
  source ~/venv/bin/activate
  # for now, reinstall bloom_inference everytime
  pip install -e bloom_inference/
else
  echo "creating venv"
  # get Linux updates, 'yes' to all
  yes | sudo apt-get update
  yes | sudo apt-get install python3.8-venv

  # create venv
  python3 -m venv ~/venv
  # activate venv
  source ~/venv/bin/activate

  # pip install packages (JAX, ray etc)
  pip install requests
  pip install "jax[tpu]>=0.2.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install git+https://github.com/younesbelkada/transformers@add_bloom_flax
  pip install ray==1.13.0 fabric dataclasses optax flax tqdm func_timeout
  pip install -e bloom_inference/
fi

# TODO: this should be it's own command.
TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368 ray start --address=$1 --resources="{\"tpu\": 1}"