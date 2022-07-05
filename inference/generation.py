import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental import PartitionSpec as P
from jax.experimental import maps
from jax.experimental.pjit import pjit

from transformers import FlaxGPTJForCausalLM, GPTJConfig
from transformers import AutoTokenizer

from inference.partitions import set_partitions

model, params = FlaxGPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", _do_init=False)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_sided = "left"

model.config.max_length = 128
model.config.num_beams = 1
model.config.do_sample = True
model.config.pad_token_id = tokenizer.pad_token

spec = set_partitions(model.params_shape_tree)

shard_params = pjit(
    model.to_bf16,
    in_axis_resources=(spec,),
    out_axis_resources=spec,
)


mesh_shape = (1, 8)
devices = np.array(jax.devices()).reshape(mesh_shape)
# create a mesh and bind names to mesh axses
mesh = maps.Mesh(devices, ("dp", "mp"))

# shard the model params
with mesh:
    params = shard_params(freeze(params))

def generate(params, input_ids, attention_mask):
    output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
    return output_ids

p_generate = pjit(generate, in_axis_resources=(spec, P("dp"), P("dp")), out_axis_resources=P("dp"))


prompt = "Reciepe for pasta with coconut:"
inputs = tokenizer([prompt] * 8, return_tensors="jax", padding="max_length", truncation=True, max_length=32) # BS = 8

with mesh:
    gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])

generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(generated_text)