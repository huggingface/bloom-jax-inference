import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental import PartitionSpec as P
from jax.experimental import maps
from jax.experimental.pjit import pjit
from flax.traverse_util import flatten_dict, unflatten_dict

from t5x import partitioning
from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState

from bloom_inference import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer

ckpt = "bigscience/bloom-6b3"

config = BloomConfig(n_layer=1)
model, params = FlaxBloomForCausalLM.from_pretrained(ckpt, _do_init=False, dtype=jnp.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-350m", use_fast=False)


# 1D parameter partitioning with 2D activation partitioning
logical_axis_rules = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    # shard remaining activations; weight matrices already have axes mapped to 'model'
    ('embed', 'model'),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]

# 2D parameter and activation partitioning
logical_axis_rules_full = [
    ('batch', 'data'),
    ('mlp', 'model'),
    ('heads', 'model'),
    ('vocab', 'model'),
    # shard both activations and weight matrices on the remaining available axis
    ('embed', 'model'),
    ('embed', 'data'),
    ('kv', None),
    ('joined_kv', None),
    ('relpos_buckets', None),
    ('abspos_buckets', None),
    ('length', None),
    ('layers', None),
    ('stack', None),
    ('mlp_activations', None),
]


# TODO: Add this in model init
def init_fn():
    input_shape = (1,1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    return model.module.init(rng, input_ids, attention_mask, return_dict=False)


param_axes = jax.eval_shape(init_fn)["params_axes"] # Axis names metadata

# create InferenceState, since the partitioner expects it. 
state = InferenceState(
    step=jnp.array(0),
    params=freeze(model.params_shape_tree),
    params_axes=freeze(param_axes),
    flax_mutables=None,
    flax_mutables_axes=param_axes,
)

num_mp_partitions = 8
partitioner = PjitPartitioner(num_mp_partitions, logical_axis_rules=logical_axis_rules_full)

mesh_axes = partitioner.get_mesh_axes(state)
params_spec = mesh_axes.params

shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)

# This will auto-magically run in mesh context
params = shard_params(freeze(params))

def generate(params, input_ids, attention_mask):
    output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
    return output_ids

p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, P("data"), P("data")),
    out_axis_resources=P("data")
)

tokenizer.padding_side = "left"
model.config.max_length = 256
model.config.num_beams = 1
model.config.do_sample = True
model.config.pad_token_id = tokenizer.pad_token_id

prompt = "Reciepe for pasta with coconut:"
inputs = tokenizer([prompt] * 8, return_tensors="jax", padding="max_length", truncation=True, max_length=64) # BS = 8

gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])
generated_text = tokenizer.batch_decode(np.asarray(gen_ids), skip_special_tokens=True)