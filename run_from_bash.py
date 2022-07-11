import functools
from multiprocessing import pool
import os

import ray
from ray_tpu import get_connection, start_ray

# script to run
bash_script = "run_sharding_example.sh"

#tpu_name = "suraj-tpu-v3-32"
tpu_name = "patrick-tpu-v3-32"
region = "europe-west4-a"

# get Python list of TPU hosts
conns = get_connection(tpu_name, region)

head_info = ray.init(include_dashboard=False, object_store_memory=10**9)
address = head_info.address_info['address']

# start ray CPU<->TPU on all hosts, copies all bash scripts and python scripts from CPU to TPU hosts
with pool.ThreadPool(processes=len(conns)) as p:
    p.map(functools.partial(start_ray, address=address), conns)

ray.shutdown()

os.system(f'gcloud alpha compute tpus tpu-vm ssh {tpu_name} --zone europe-west4-a --worker="all" --command="bash bloom_inference/{bash_script}"')

