import functools
from  multiprocessing import pool

import ray
from ray_tpu import create_tpu, wait_til, get_connection, start_ray

from inference.tpu_cluster import TPUCluster

cores_per_replica = 8
tpu_size = 32

tpu_name = "suraj-tpu-v3-32"
region = "europe-west4-a"

# conns = get_connection(tpu_name, region)


head_info = ray.init(include_dashboard=False, object_store_memory=10**9)
address = head_info['redis_address']

# with pool.ThreadPool(processes=len(conns)) as p:
    # p.map(functools.partial(start_ray, address=address), conns)

start_ray(conns, address=address)

t = TPUCluster((tpu_size // cores_per_replica, cores_per_replica), len(conns))