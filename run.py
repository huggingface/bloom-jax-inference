import functools
import time
from multiprocessing import pool

import ray
from ray_tpu import get_connection, start_ray

from bloom_inference.tpu_manager import TPUManager 

tpu_name="bloom-tpu-v4-64"
region="us-central2-b"

ckpt = "bigscience/bloom"
t5x_path = "gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x/checkpoint_0"
max_len = 128
max_input_len = 64
model_parallel_submesh = (1, 2, 4, 1) # for v4-64


def setup():
    # get Python list of TPU hosts
    conns = get_connection(tpu_name, region)
    print(len(conns))
    address='10.130.0.10:8080'
    head_info = ray.init(include_dashboard=False, address="auto")
    # object_store_memory=10**9, 

    # start ray CPU<->TPU on all hosts
    with pool.ThreadPool(processes=len(conns)) as p:
        p.map(functools.partial(start_ray, address=address), conns)

def init_manager():
    # initialise TPU manager
    t = TPUManager(
        8,
        ckpt=ckpt,
        t5x_path=t5x_path,
        max_len=max_len,
        max_input_len=max_input_len,
        model_parallel_submesh=model_parallel_submesh,
    )
    return t

# # benchmark compile step
# start = time.time()
# print(t.generate(4*['Recipe for coconut pasta:']))
# print(f"Generations completed in ***REMOVED***time.time() - start:.06***REMOVED***s")

# # benchmark generate
# start = time.time()
# print(t.generate(4*['Recipe for coconut pasta:']))
# print(f"Generations completed in ***REMOVED***time.time() - start:.06***REMOVED***s")

# # shutdown ray rpc
# ray.shutdown()
