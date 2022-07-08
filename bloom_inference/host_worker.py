import ray
import time
from queue import Queue


@ray.remote(resources={"tpu": 1})
class TPUHostWorker(object):
    def __init__(self, num_mp_partitions):
        # TODO: add other generation hp's as attributes
        self.num_mp_partitions = num_mp_partitions

        self.input_q = Queue(maxsize=1)
        self.output_q = Queue(maxsize=1)

    def run(self):
        # we import packages here to import JAX and Generator only on the Host worker and not the CPU manager
        import jax
        from bloom_inference.generator import Generator, head_print

        print(f"jax runtime initialization starting")
        start = time.time()
        head_print(f"jax devices: {jax.device_count()}")
        head_print(f"jax runtime initialized in {time.time() - start:.06}s")

        # load model and params
        head_print("Loading model")
        generator = Generator(self.num_mp_partitions)
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
