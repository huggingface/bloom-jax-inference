from fastapi import FastAPI, Request

import functools
from multiprocessing import pool

import ray
from ray_tpu import get_connection, start_ray
from bloom_inference.tpu_manager import TPUManager

tpu_name = "dalle-pod"
region = "us-east1-d"

ckpt = "bigscience/bloom"
t5x_path = "gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x-final/checkpoint_0"

# IGNORED
max_len = 128
max_input_len = 64

model_parallel_submesh = (1, 8, 1, 2)  # for v3-256

# get Python list of TPU host
conns = get_connection(tpu_name, region)

head_info = ray.init(include_dashboard=False, object_store_memory=10 ** 9)
address = head_info.address_info['address']


# start ray CPU<->TPU on all hosts
with pool.ThreadPool(processes=len(conns)) as p:
    p.map(functools.partial(start_ray, address=address), conns)

# initialise TPU manager
tpu_manager = TPUManager(
        len(conns),
        ckpt=ckpt,
        t5x_path=t5x_path,
        max_len=max_len,
        max_input_len=max_input_len,
        model_parallel_submesh=model_parallel_submesh,
)

print("Compiling greedy...")
inputs = "up" * 128
tpu_manager.generate(inputs, do_sample=False)

print("Compiling sampling...")
tpu_manager.generate(inputs, do_sample=True)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate/")
async def generate(request: Request):
    content = await request.json()
    inputs = content.get("inputs", "Hello my name is BLOOM")
    do_sample = content.get("do_sample", "True")
    generation = tpu_manager.generate(inputs, do_sample)[0]
    out = [{"generated_text": generation[0]}]
    return out
