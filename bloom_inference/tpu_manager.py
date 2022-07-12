import time
import ray
import numpy as np

class TPUManager:
    # @func_set_timeout(1200)
    def __init__(self, num_mp_partitions, node_count):
        # needs a valid ray cluster to start
        assert ray.is_initialized(), "ray not initialised"

        from bloom_inference.host_worker import TPUHostWorker

        self.nodes = []
        self.node_count = node_count

        start = time.time()

        for i in range(node_count):
            self.nodes.append(TPUHostWorker.options(max_concurrency=2).remote(num_mp_partitions))

        for node in self.nodes:
            node.run.remote()

        print(f"TPU workers created in ***REMOVED***time.time() - start:.06***REMOVED***s")


    # @func_set_timeout(600)
    def generate(self, context):
        # TODO: split context (prompts) if len(context) != 4
        #context = np.array_split(context, len(self.nodes), axis=0)
        res = []

        for n, ctx in zip(self.nodes, context):
            res.append(n.generate.remote(ctx))

        return [i for i in ray.get(res)]
