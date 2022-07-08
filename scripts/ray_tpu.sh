#!/usr/bin/env bash
set -e

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c "import time; time.sleep(999999999)"

# debugging
rm -rf ~/venv
rm -rf ~/t5x

# check if venv exists
if [ -f ~/venv/bin/activate ];
then
  echo "venv exists"
  # activate venv (if not done so already)
  source ~/venv/bin/activate
  # for now, reinstall bloom_inference everytime
  pip install -e bloom_inference/
else
  echo "creating venv"
  # get application updates, 'yes' to all
  yes | sudo apt-get update
  yes | sudo apt-get install python3.8-venv

  # create venv
  python3 -m venv ~/venv
  # activate venv
  source ~/venv/bin/activate

  # pip install standard packages
  pip install ray==1.13.0 transformers fabric dataclasses tqdm func_timeout
  # build T5X from source
  git clone --branch=main https://github.com/google-research/t5x
  cd t5x && pip install -e '.[tpu]' -f \https://storage.googleapis.com/jax-releases/libtpu_releases.html && cd ..
  rm -rf ~/.cache
  pip uninstall jax
  pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  # And finally, Flax BLOOM
  pip install -e bloom_inference/
fi

# TODO: this should be it's own command.
TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368 ray start --address=$1 --resources="{\"tpu\": 1}"