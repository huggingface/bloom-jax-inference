import time
import ray
import numpy as np

class TPUManager:
    # @func_set_timeout(1200)
    def __init__(self,
                 mesh_shape,
                 node_count):
        assert ray.is_initialized()  # needs a valid ray cluster to start

        from bloom_inference.host_worker import TPUHostWorker

        self.nodes = []
        self.node_count = node_count

        start = time.time()

        for i in range(node_count):
            self.nodes.append(TPUHostWorker.options(max_concurrency=2).remote(mesh_shape))

        for node in self.nodes:
            node.run.remote()

        print(f"TPU workers created in {time.time() - start:.06}s")


    # @func_set_timeout(600)
    def generate(self, context):
        start = time.time()
        #context = np.array_split(context, len(self.nodes), axis=0)
        res = []

        for n, ctx in zip(self.nodes, context):
            res.append(n.generate.remote(ctx))

        return [i for i in ray.get(res)], f"Generations completed in {time.time() - start:.06}s"
