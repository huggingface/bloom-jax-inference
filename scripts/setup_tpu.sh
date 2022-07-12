#!/usr/bin/env bash
set -e

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python -c "import time; time.sleep(999999999)"

# debugging
#rm -rf ~/venv
#rm -rf ~/t5x
#rm -rf ~/.cache/pip

if [ -f ~/bloom-jax-inference]
then
    pushd ~/bloom-jax-inference
    git pull
    popd
else
    git clone -b v3-pod https://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@github.com/huggingface/bloom-jax-inference
fi

gcloud auth activate-service-account --key-file ~/bloom-jax-inference/key.json
export GOOGLE_APPLICATION_CREDENTIALS=~/bloom-jax-inference/key.json

# check if venv exists
if [ -f ~/venv/bin/activate ];
then
    echo "venv exists"
    # activate venv (if not done so already)
    source ~/venv/bin/activate
    # for now, reinstall bloom-jax-inference everytime
    pip install -e bloom-jax-inference/
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
    pip install -U pip
    pip install ray==1.13.0 transformers fabric dataclasses tqdm func_timeout
    
    # build T5X from source
    git clone --branch=main https://github.com/google-research/t5x
    cd t5x
    python3 -m pip install -e '.[tpu]' -f \https://storage.googleapis.com/jax-releases/libtpu_releases.html
    cd ..
    
    rm -rf ~/.cache/pip

    # force JAX to TPU version
    yes | pip uninstall jax jaxlib
    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    
    # And finally, Flax BLOOM
    pip install -e bloom-jax-inference/
fi
