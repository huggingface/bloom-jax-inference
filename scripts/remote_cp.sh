INSTANCE=bloom-tpu-v4-64
ZONE=us-central2-b
PROJECT="huggingface-ml"

echo copying $1 to $2

for n in {1..5}; do
    gcloud alpha compute tpus tpu-vm scp $1 $INSTANCE:$2 --project=$PROJECT --zone=$ZONE --force-key-file-overwrite --strict-host-key-checking=no --worker=all
    sleep 1
done