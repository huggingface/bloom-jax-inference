import functools
import time
from multiprocessing import pool

import ray
from ray_tpu import get_connection, start_ray

from bloom_inference.tpu_manager import TPUManager 

cores_per_replica = 8
tpu_size = 32

#tpu_name = "suraj-tpu-v3-32"
tpu_name = "patrick-tpu-v3-32"
region = "europe-west4-a"

conns = get_connection(tpu_name, region)

head_info = ray.init(include_dashboard=False, object_store_memory=10**9)
address = head_info.address_info['address']

with pool.ThreadPool(processes=len(conns)) as p:
    p.map(functools.partial(start_ray, address=address), conns)

t = TPUManager((tpu_size // cores_per_replica, cores_per_replica), len(conns))

# benchmark compile step
start = time.time()
t.generate(4*['the cat sat on the'])
print(f"Generations completed in {time.time() - start:.06}s")

# benchmark generate
start = time.time()
t.generate(4*['the cat sat on the'])
print(f"Generations completed in {time.time() - start:.06}s")

conns.run("ray stop -f")
ray.shutdown()