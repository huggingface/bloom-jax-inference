import functools
import time
from multiprocessing import pool

import ray
from ray_tpu import get_connection, start_ray

from bloom_inference.tpu_manager import TPUManager 

tpu_name = "dalle-pod"
region = "us-east1-d"
project = "dall-e-mega"

ckpt = "bigscience/bloom"

t5x_path = "gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x-final/checkpoint_0"

batch_size = 16

max_new_tokens = 64
max_input_len = 64
max_len = max_input_len + max_new_tokens

model_parallel_submesh = (1, 8, 1, 2)  # for v3-256

benchmark = True

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
print(t.generate(batch_size*['Recipe for coconut pasta:']))
print(f"Generations completed in {time.time() - start:.06}s")

# benchmark generate
start = time.time()
print(t.generate(batch_size*['Recipe for coconut pasta:']))
print(f"Generations completed in {time.time() - start:.06}s")


if benchmark:
    for i in [1, 2, 4, 8, 16, 32, 64]:
        # benchmark compile step
        t.generate(i*['Recipe for coconut pasta:'])

        # benchmark generate
        start = time.time()
        t.generate(i*['Recipe for coconut pasta:'])
        print(f"{time.time() - start:.06}s")

# shutdown ray rpc
ray.shutdown()
