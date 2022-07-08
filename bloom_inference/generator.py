import warnings

import jax
import numpy as np
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
from jax.experimental import maps
from jax.experimental.pjit import pjit
from jax.experimental.compilation_cache import compilation_cache as cc

from transformers import AutoTokenizer, FlaxBloomForCausalLM

from bloom_inference.partitions import set_partitions

cc.initialize_cache("~/jax_cache")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)

if jax.process_index() == 0:
    warnings.filterwarnings("default")


# print but only on the first node
def head_print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)

class Generator:
    def __init__(self, mesh_shape, ckpt="bigscience/bloom-6b3"):
        # create a mesh and bind names to mesh axes
        self.mesh_shape = mesh_shape
        self.devices = np.array(jax.devices()).reshape(self.mesh_shape)

        self.ckpt = ckpt

    def load_model_and_params(self):
        if self.ckpt.split("-")[-1] == "scan":
            use_scan = True
        else:
            use_scan = False

        # TODO loading params should be done in a thread
        model, self.params = FlaxBloomForCausalLM.from_pretrained("sanchit-gandhi/bloom-6b3-scan", _do_init=False, use_scan=True)
        self.spec = set_partitions(model.params_shape_tree)

        tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        # setup for generation
        tokenizer.padding_sided = "left"
        model.config.max_length = 64
        model.config.num_beams = 1
        model.config.do_sample = True

        self.model = model
        self.tokenizer = tokenizer

        self.p_shard_params = pjit(
            self.model.to_bf16,
            in_axis_resources=(self.spec,),
            out_axis_resources=self.spec,
        )

        def generate(params, input_ids, attention_mask):
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
            return output_ids
        
        self.p_generate = pjit(generate, in_axis_resources=(self.spec, P("dp"), P("dp")), out_axis_resources=P("dp"))

    def shard_params(self):
        with maps.Mesh(self.devices, ("dp", "mp")):
            self.params = self.p_shard_params(freeze(self.params))

    def generate(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="jax", padding="max_length", truncation=True, max_length=16) # BS = 4
        with maps.Mesh(self.devices, ("dp", "mp")):
            gen_ids = self.p_generate(freeze(self.params), inputs["input_ids"], inputs["attention_mask"])

        generated_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return generated_text
