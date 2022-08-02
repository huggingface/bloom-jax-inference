INSTANCE=sanchit-tpu-v3-32
ZONE=europe-west4-a
PROJECT="huggingface-ml"

# run script.bash through run_script.bash
gcloud alpha compute tpus tpu-vm ssh $INSTANCE --project=$PROJECT --zone=$ZONE \
    --force-key-file-overwrite --strict-host-key-checking=no \
    --worker=all \
    --command="source ~/venv/bin/activate && python ~/bloom-jax-inference/profiling_example.py"

# source ~/venv/bin/activate && python ~/bloom-jax-inference/profiling_example.py