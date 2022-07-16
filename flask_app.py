from typing import Union

from flask import Flask, jsonify, make_response, request

import functools
import json
import time
import threading
import datetime
from multiprocessing import pool
from queue import Queue, Empty

import ray
from ray_tpu import get_connection, start_ray

from bloom_inference.tpu_manager import TPUManager 

tpu_name="bloom-tpu-v4-64"
region="us-central2-b"

# ckpt = "bigscience/bloom-6b3"
# t5x_path = "gs://bloom-jax-us-central2-b/bloom-6b3-scan-t5x-v3-8-pretrained/checkpoint_0"
ckpt = "bigscience/bloom"
# t5x_path = "gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x/checkpoint_0"
t5x_path = "gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x-final/checkpoint_0"

MAX_NEW_TOKENS= 64
INPUT_LENGTH = 64
BATCH_SIZE=1
QUEUE_SIZE=1

max_len = INPUT_LENGTH + MAX_NEW_TOKENS
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
        max_input_len=INPUT_LENGTH,
        model_parallel_submesh=model_parallel_submesh,
    )
    return t





def precompile(tpu_manager):
    # compile generate
    start = datetime.datetime.now()
    print("Compiling sampling")
    inputs = BATCH_SIZE * ["This is a cat"]
    tpu_manager.generate(inputs, True)
    print(f"Compiled sampling in {datetime.datetime.now() - start}")

    start = datetime.datetime.now()
    print("Compiling greedy")
    tpu_manager.generate(inputs, False)
    print(f"Compiled greedy in {datetime.datetime.now() - start}")


def run_app(q):
    app = Flask(__name__)

    @app.route("/generate", methods=["POST"])
    def generate():
        content = request.json

        if not content.get("parameters", {}).get("do_sample", False):
            return make_response({"error": "Not accepting greedy atm."}, 500)

        qsize = q.qsize()
        if qsize >= QUEUE_SIZE:
            return make_response({"error": "queue full try again later"}, 503)

        print(f"Size of queue is {qsize}")

        response_queue = Queue()
        q.put((content.get("inputs", "Hello my name is bloom"), response_queue))

        out = response_queue.get()

        print("Received from queue")

        return make_response(jsonify([{"generated_text": out[:]}]), 200)

    app.run(port=8000, host="127.0.0.1")

def server_loop(q):
    print("Starting server loop")

    #tpu_name = "suraj-tpu-v3-32"
    tpu_name = "patrick-tpu-v3-32"
    region = "europe-west4-a"

    setup()
    t = init_manager()

    precompile(t)

    try:
        while True:
            # benchmark generate
            print("Waiting for queries")
            (prompt, response_queue) = q.get()
            all_prompts = [prompt]
            all_queues = [response_queue]
            while len(all_prompts) < BATCH_SIZE:
                try:
                    (prompt, response_queue) = q.get(block=False)
                    all_prompts.append(prompt)
                    all_queues.append(response_queue)
                except Empty:
                    break

            # Fill in the batch if not present
            # Not appending queues since no one is listening for 
            # those results
            for i in range(len(all_prompts), BATCH_SIZE):
                all_prompts.append('')

            start = time.time()
            results = t.generate(all_prompts, True)[0]
            print(f"Generations completed in {time.time() - start:.06}s")

            for result, response_queue in zip(results, all_queues):
                response_queue.put(result)
    finally:
        ray.shutdown()

if __name__ == "__main__":
    q = Queue()

    threading.Thread(target=run_app, args=(q, )).start()
    server_loop(q)
