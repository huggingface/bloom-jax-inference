import time
import ray
import numpy as np

class TPUManager:
    # @func_set_timeout(1200)
    def __init__(
        self,
        node_count=8,
        ckpt="bigscience/bloom",
        t5x_path="gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x-final/checkpoint_0",
        max_len=256,
        max_input_len=64,
        model_parallel_submesh=(1, 8, 1, 2),  # for v3-256
        num_mp_partitions=None,

    ):
        # needs a valid ray cluster to start
        assert ray.is_initialized(), "ray not initialised"

        from bloom_inference.host_worker import TPUHostWorker

        self.ckpt = ckpt
        self.path = t5x_path
        self.max_len = max_len
        self.max_input_len = max_input_len
        self.model_parallel_submesh = model_parallel_submesh
        self.num_mp_partitions = num_mp_partitions

        self.nodes = []
        self.node_count = node_count

        start = time.time()

        for i in range(node_count):
            worker = TPUHostWorker.options(max_concurrency=2).remote(
                ckpt,
                t5x_path,
                max_len,
                max_input_len,
                model_parallel_submesh,
                num_mp_partitions,
            )
            self.nodes.append(worker)

        for node in self.nodes:
            node.run.remote()

        print(f"TPU workers created in {time.time() - start:.06}s")


    # @func_set_timeout(600)
    def generate(self, inputs, do_sample=True):
        res = []

        for n in self.nodes:
            res.append(n.generate.remote(inputs, do_sample))

        res = ray.get(res)

        return [i for i in res]
