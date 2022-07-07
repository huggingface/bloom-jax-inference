import warnings

import jax
import numpy as np
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental import PartitionSpec as P
from jax.experimental import maps
from jax.experimental.pjit import pjit
from jax.experimental.compilation_cache import compilation_cache as cc

from transformers import FlaxGPTJForCausalLM, GPTJConfig
from transformers import AutoTokenizer

from bloom_inference.partitions import set_partitions

cc.initialize_cache("~/jax_cache")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)

if jax.host_id() == 0:
    warnings.filterwarnings("default")


# print but only on the first node
def head_print(*args, **kwargs):
    if jax.host_id() == 0:
        print(*args, **kwargs)

class Generator:
    def __init__(self, mesh_shape, ckpt="EleutherAI/gpt-j-6B"):
        # create a mesh and bind names to mesh axses
        self.mesh_shape = mesh_shape
        self.devices = np.array(jax.devices()).reshape(self.mesh_shape)
        # self.mesh = maps.Mesh(devices, ("dp", "mp"))

        # load the model and params
        # self.load_model_and_params()

    def load_model_and_params(self):
        # TODO loading params should be done in a thread
        model, self.params = FlaxGPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", _do_init=False)
        self.spec = set_partitions(model.params_shape_tree)

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        # setup for generation
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_sided = "left"
        model.config.max_length = 128
        model.config.num_beams = 1
        model.config.do_sample = True
        model.config.pad_token_id = tokenizer.pad_token_id

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
        inputs = self.tokenizer(prompts, return_tensors="jax", padding="max_length", truncation=True, max_length=32) # BS = 8
        with maps.Mesh(self.devices, ("dp", "mp")):
            gen_ids = self.p_generate(freeze(self.params), inputs["input_ids"], inputs["attention_mask"])

        generated_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return generated_text