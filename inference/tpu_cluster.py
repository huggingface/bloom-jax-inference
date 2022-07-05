import itertools
import json
import time

import ray

import numpy as np

from .runner import NetworkRunner
# from func_timeout import func_set_timeout


class TPUCluster:
    # @func_set_timeout(1200)
    def __init__(self,
                 mesh_shape,
                 node_count):
        assert ray.is_initialized()  # needs a valid ray cluster to start
        self.nodes = []
        self.node_count = node_count
        self.dp, self.mp = mesh_shape

        start = time.time()

        for i in range(node_count):
            self.nodes.append(NetworkRunner.options(max_concurrency=2).remote(mesh_shape))

        for n in self.nodes:
            n.run.remote()

        params = []
        for n in self.nodes:
            params.append(n.get_params.remote())

        self.param_count = ray.get(params)[0]
        print(f"Ray actors created in ***REMOVED***time.time() - start:.06***REMOVED***s")


    # @func_set_timeout(600)
    def generate(self, context):
        context = np.array_split(context, len(self.nodes), axis=0)
        res = []
        for n, ctx in zip(self.nodes, context):
            res.append(n.generate.remote(ctx))

        return [i for i in ray.get(res)]
