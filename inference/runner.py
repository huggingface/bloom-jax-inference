# copied from https://github.com/kingoflolz/mesh-trzansformer-jax

import ray
import time
import jax
import numpy as np
from queue import Queue



# print but only on the first node
def head_print(*args, **kwargs):
    if jax.host_id() == 0:
        print(*args, **kwargs)


@ray.remote(resources=***REMOVED***"tpu": 1***REMOVED***)
class NetworkRunner(object):
    def __init__(self, mesh_shape):
        self.mesh_shape = mesh_shape

        self.input_q = Queue(maxsize=1)
        self.output_q = Queue(maxsize=1)

    def run(self):
        print(f"jax runtime initialization starting")
        import jax
        import jax.numpy as jnp
        from flax.core.frozen_dict import freeze, unfreeze
        from jax.experimental import PartitionSpec as P
        from jax.experimental import maps
        from jax.experimental.pjit import pjit

        from transformers import FlaxGPTJForCausalLM, GPTJConfig
        from transformers import AutoTokenizer

        from inference.partitions import set_partitions

        start = time.time()
        jax.devices()

        import warnings
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=ResourceWarning)

        if jax.host_id() == 0:
            warnings.filterwarnings("default")

        head_print(f"jax devices: ***REMOVED***jax.device_count()***REMOVED***")
        head_print(f"jax runtime initialized in ***REMOVED***time.time() - start:.06***REMOVED***s")
        
        # load model and params here
        model, params = FlaxGPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", _do_init=False)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

        # setup for generation
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_sided = "left"
        model.config.max_length = 128
        model.config.num_beams = 1
        model.config.do_sample = True
        model.config.pad_token_id = tokenizer.pad_token

        def generate(params, input_ids, attention_mask):
            output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
            return output_ids

        # crreate the partition spec
        spec = set_partitions(model.params_shape_tree)
        p_generate = pjit(generate, in_axis_resources=(spec, P("dp"), P("dp")), out_axis_resources=P("dp"))

        shard_params = pjit(
            model.to_bf16,
            in_axis_resources=(spec,),
            out_axis_resources=spec,
        )

        # create a mesh and bind names to mesh axses
        devices = np.array(jax.devices()).reshape(self.mesh_shape)
        mesh = maps.Mesh(devices, ("dp", "mp"))

        with mesh:
            start = time.time()
            params = shard_params(freeze(params))
            head_print(f"Initialized in ***REMOVED***time.time() - start:.06***REMOVED***s")

            while True:
                operation, prompts = self.input_q.get()
                if operation == "generate":
                    inputs = tokenizer(prompts, return_tensors="jax", padding="max_length", truncation=True, max_length=32) # BS = 8
                    gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])
                    generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    self.output_q.put(generated_text)
                else:
                    raise Exception("Not implemented")

    def get_params(self):
        self.input_q.put(("get_params", None))
        return self.output_q.get()

    def generate(self, input):
        self.input_q.put(("generate", input))
        return self.output_q.get()
