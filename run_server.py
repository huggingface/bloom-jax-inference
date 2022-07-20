import functools
import time
import threading
from multiprocessing import pool

import ray
from ray_tpu import get_connection, start_ray
from queue import Queue, Empty

from bloom_inference.tpu_manager import TPUManager
from flask import Flask, jsonify, make_response, request


BATCH_SIZE = 2
QUEUE_SIZE = 32

MAX_NEW_TOKENS = 16
MAX_INPUT_LEN = 16
MAX_LEN = MAX_INPUT_LEN + MAX_NEW_TOKENS

def run_app(q):
    app = Flask(__name__)

    @app.route("/generate", methods=["POST"])
    def generate():
        content = request.json

        qsize = q.qsize()
        if qsize >= QUEUE_SIZE:
            return make_response({"error": "queue full try again later"}, 503)

        response_queue = Queue()
        parameters = content.get("parameters", {})
        do_sample = parameters.get("do_sample", False)
        q.put((content.get("inputs", "Hello my name is bloom"), response_queue, do_sample))

        out = response_queue.get()

        return make_response(jsonify([{"generated_text": out}]), 200)

    app.run(port=8000, host="127.0.0.1")


def server_loop(q):
    print("Starting server loop")
    tpu_name = "patrick-tpu-v3-32"
    region = "europe-west4-a"

    ckpt = "bigscience/bloom"

    t5x_path = "gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x-final/checkpoint_0"

    num_mp_partitions = 4

    conns = get_connection(tpu_name, region)

    head_info = ray.init(include_dashboard=False, object_store_memory=10 ** 9)
    try:
        address = head_info.address_info['address']

        with pool.ThreadPool(processes=len(conns)) as p:
            p.map(functools.partial(start_ray, address=address), conns)

        t = TPUManager(
            len(conns),
            ckpt=ckpt,
            t5x_path=t5x_path,
            max_len=MAX_LEN,
            max_input_len=MAX_INPUT_LEN,
            num_mp_partitions=num_mp_partitions,
        )

        # benchmark compile step
        start = time.time()
        print(t.generate(BATCH_SIZE * ['the cat sat on the']))
        print(f"Generations compiled in {time.time() - start:.06}s")

        while True:
            # benchmark generate
            print("Waiting for queries")
            (prompt, response_queue, do_sample) = q.get()
            all_prompts = [prompt]
            all_queues = [response_queue]
            while len(all_prompts) < BATCH_SIZE:
                try:
                    (prompt, response_queue, do_sample) = q.get(block=False)
                    all_prompts.append(prompt)
                    all_queues.append(response_queue)
                except Empty:
                    break

            # Fill in the batch if not present
            for i in range(len(all_prompts), BATCH_SIZE):
                all_prompts.append('')

            start = time.time()
            # TODO: handle `do_sample` -> currently always set to True
            results = t.generate(all_prompts, do_sample=True)
            print(f"Generations completed in {time.time() - start:.06}s")

            for result, response_queue in zip(results, all_queues):
                response_queue.put(result)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    q = Queue()

    threading.Thread(target=run_app, args=(q,)).start()
    server_loop(q)
