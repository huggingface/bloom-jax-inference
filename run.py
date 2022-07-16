import functools
import json
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
    # head_info = ray.init(include_dashboard=False, address="auto")
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

# def benchmark_generate(tpu_manager, inputs, bs):
#     print(f"benchmarking for BS = {bs}")
#     # benchmark compile step
#     start = time.time()
#     tpu_manager.generate(inputs)
#     print(f"Generations compiled in {time.time() - start:.06}s")

#     # benchmark generate
#     start = time.time()
#     out = tpu_manager.generate(inputs)
#     with open(f'result-{bs}.json', 'w') as fp:
#         json.dump(out[0], fp)
#     print(f"Generations completed in {time.time() - start:.06}s")
#     print("=========================================")


# setup()
# tpu_manager = init_manager()

# p = [
#     "Question: Where does the Greek Goddess Persephone spend half of the year when she is not with her mother? Answer:",
#     "spelling test answers. What are the letters in « language »? Answer: l-a-n-g-u-a-g-e What are the letters in « Romanian »? Answer:",
#     "Math exercise - answers: 34+10=44 54+20=",
#     "A poem about the beauty of science by Alfred Edgar Brittle Title: The Magic Craft In the old times"
# ]


# inputs = {"prompt": p[0], "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)

# inputs = {"prompt": p[:2], "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 2)

# inputs = {"prompt": p, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 4)

# inputs = {"prompt": p * 2, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 8)

# inputs = {"prompts": "one" * 64, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)

# inputs = {"prompts": "one" * 128, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)

# inputs = {"prompts": "one" * 192, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)

# inputs = {"prompts": "one" * 256, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)


# inputs = {"prompts": "one" * 64, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)

# inputs = {"prompts": "one" * 256, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)

# inputs = {"prompts": "one" * 128, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)

# inputs = {"prompts": "one" * 64, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)

# inputs = {"prompts": "one" * 128, "max_new_tokens": 64, "gen_method": "sampling"}
# benchmark_generate(tpu_manager, inputs, 1)


# shutdown ray rpc
ray.shutdown()

# Generations compiled in 329.895s
# Generations completed in 1.30133s
# =========================================
# benchmarking for BS = 2
# Generations compiled in 70.9726s
# Generations completed in 12.2895s
# =========================================
# benchmarking for BS = 4
# Generations compiled in 72.5203s
# Generations completed in 12.4379s
# =========================================
# benchmarking for BS = 8
# Generations compiled in 27.7761s
# Generations completed in 12.6601s


## 64 new tokens
# Generations compiled in 245.494s
# Generations completed in 2.22431s
# =========================================
# benchmarking for BS = 2
# Generations compiled in 78.9155s
# Generations completed in 24.5159s
# =========================================
# benchmarking for BS = 4
# Generations compiled in 93.1491s
# Generations completed in 24.7582s
# =========================================
# benchmarking for BS = 8
# Generations compiled in 40.2777s
# Generations completed in 25.0843s
# =========================================


## 64 + 64
# Generations compiled in 245.809s
# Generations completed in 2.23594s
# =========================================
# 128 + 64
# benchmarking for BS = 1
# Generations compiled in 55.2179s
# Generations completed in 2.2969s
# =========================================
# 192 + 64
# benchmarking for BS = 1
# Generations compiled in 72.6207s
# Generations completed in 2.31498s
# =========================================
#  256 + 64
# benchmarking for BS = 1
# Generations compiled in 68.7094s
# Generations completed in 2.40084s
# =========================================