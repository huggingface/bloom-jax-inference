import functools
import time
from multiprocessing import pool

import ray
from ray_tpu import get_connection, start_ray

from bloom_inference.tpu_manager import TPUManager 

num_mp_partitions = 8

# tpu_name = "suraj-tpu-v3-32"
# tpu_name = "patrick-tpu-v3-32"
# region = "europe-west4-a"
tpu_name = "dalle-pod"
region = "us-east1-d"
project = "dall-e-mega"

ckpt = "bigscience/bloom-6b3"
# t5x_path = "gs://suraj-tpu-bucket/bloom-6b3-scan-t5x-v3-8-pretrained/checkpoint_0"
t5x_path = "gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x/checkpoint_0"
max_len = 256
max_input_len = 64
# model_parallel_submesh = (1, 16, 1, 2)  # for v3-256
model_parallel_submesh = (1, 2, 4, 1), # for v4-64


# get Python list of TPU hosts
conns = get_connection(tpu_name, region)

head_info = ray.init(include_dashboard=False, object_store_memory=10**9)
address = head_info.address_info['address']

# start ray CPU<->TPU on all hosts
with pool.ThreadPool(processes=len(conns)) as p:
    p.map(functools.partial(start_ray, address=address), conns)

# initialise TPU manager
t = TPUManager(
    len(conns),
    ckpt=ckpt,
    t5x_path=t5x_path,
    max_len=max_len,
    max_input_len=max_input_len,
    model_parallel_submesh=model_parallel_submesh,
)

# benchmark compile step
start = time.time()
print(t.generate(4*['Recipe for coconut pasta:']))
print(f"Generations completed in {time.time() - start:.06}s")

# benchmark generate
start = time.time()
print(t.generate(4*['Recipe for coconut pasta:']))
print(f"Generations completed in {time.time() - start:.06}s")

# shutdown ray rpc
ray.shutdown()
