# copied from https://github.com/kingoflolz/mesh-trzansformer-jax

import ray
import time
import numpy as np
from queue import Queue


@ray.remote(resources={"tpu": 1})
class TPUHostWorker(object):
    def __init__(self, mesh_shape):
        self.mesh_shape = mesh_shape

        self.input_q = Queue(maxsize=1)
        self.output_q = Queue(maxsize=1)

    def run(self):
        print(f"jax runtime initialization starting")
        import jax
        
        from bloom_inference.generator import Generator, head_print

        start = time.time()
        jax.devices()
        head_print(f"jax devices: {jax.device_count()}")
        head_print(f"jax runtime initialized in {time.time() - start:.06}s")
        
        # load model and params here
        head_print("Loading model")
        generator = Generator(self.mesh_shape)
        generator.load_model_and_params()
        head_print("Loading complete")

        start = time.time()
        generator.shard_params()
        head_print(f"Initialized in {time.time() - start:.06}s")

        while True:
            operation, prompts = self.input_q.get()
            if operation == "generate":
                generated_text = generator.generate(prompts)
                self.output_q.put(generated_text)
            else:
                raise Exception("Not implemented")

    def generate(self, input):
        self.input_q.put(("generate", input))
        return self.output_q.get()
