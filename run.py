import functools
import time
from multiprocessing import pool

import ray
from ray_tpu import get_connection, start_ray

from bloom_inference.tpu_manager import TPUManager 

num_mp_partitions = 8

#tpu_name = "suraj-tpu-v3-32"
tpu_name = "patrick-tpu-v3-32"
region = "europe-west4-a"

# get Python list of TPU hosts
conns = get_connection(tpu_name, region)

head_info = ray.init(include_dashboard=False, object_store_memory=10**9)
address = head_info.address_info['address']

# start ray CPU<->TPU on all hosts
with pool.ThreadPool(processes=len(conns)) as p:
    p.map(functools.partial(start_ray, address=address), conns)

# initialise TPU manager
t = TPUManager(num_mp_partitions, len(conns))

# benchmark compile step
start = time.time()
print(t.generate(4*['Recipe for coconut pasta:']))
print(f"Generations completed in ***REMOVED***time.time() - start:.06***REMOVED***s")

# benchmark generate
start = time.time()
print(t.generate(4*['Recipe for coconut pasta:']))
print(f"Generations completed in ***REMOVED***time.time() - start:.06***REMOVED***s")

# shutdown ray rpc
ray.shutdown()
