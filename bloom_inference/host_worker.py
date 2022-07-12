import ray
import time
from queue import Queue


@ray.remote(resources={"tpu": 1})
class TPUHostWorker(object):
    def __init__(
        self,
        ckpt="bigscience/bloom-6b3",
        t5x_path="gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x/checkpoint_0",
        max_len=256,
        max_input_len=64,
        model_parallel_submesh=(1, 2, 4, 1), # for v4-64
    ):
        self.ckpt = ckpt
        self.path = t5x_path
        self.max_len = max_len
        self.max_input_len = max_input_len
        self.model_parallel_submesh = model_parallel_submesh

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
        generator = Generator(
            model_parallel_submesh=self.model_parallel_submesh,
            ckpt=self.ckpt,
            t5x_path=self.path,
            max_len=self.max_len,
            max_input_len=self.max_input_len
        )
        generator.load_model_and_params()
        head_print("Loading complete")

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
