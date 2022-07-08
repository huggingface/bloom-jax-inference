import functools
import time
from multiprocessing import pool

import ray
from ray_tpu import get_connection, start_ray, stop_ray

from bloom_inference.tpu_manager import TPUManager 

cores_per_replica = 8
tpu_size = 32

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
t = TPUManager((tpu_size // cores_per_replica, cores_per_replica), len(conns))

# benchmark compile step
start = time.time()
print(t.generate(4*['Recipe for coconut pasta:']))
print(f"Generations completed in {time.time() - start:.06}s")

# benchmark generate
start = time.time()
print(t.generate(4*['Recipe for coconut pasta:']))
print(f"Generations completed in {time.time() - start:.06}s")

# shutdown TPU hosts
with pool.ThreadPool(processes=len(conns)) as p:
    p.map(stop_ray, conns)

# shutdown CPU host
ray.shutdown()
