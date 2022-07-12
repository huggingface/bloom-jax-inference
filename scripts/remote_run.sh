#!/bin/bash

INSTANCE=bloom-tpu-v4-64
ZONE=us-central2-b

# INSTANCE=$1
COMMAND=$1
PROJECT=huggingface-ml
ZONE=us-central2-b

# run script.bash through run_script.bash
gcloud alpha compute tpus tpu-vm ssh $INSTANCE --project=$PROJECT --zone=$ZONE \
    --force-key-file-overwrite --strict-host-key-checking=no \
    --worker=all \
    --command="source ~/venv/bin/activate && $COMMAND"