import time

import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState

from bloom_inference.modeling_bloom.modeling_bloom import FlaxBloomForCausalLM
from bloom_inference.modeling_bloom.configuration_bloom import BloomConfig
from transformers import AutoTokenizer

# ckpt = "sanchit-gandhi/bloom-6b3-scan-t5x"

config = BloomConfig.from_pretrained("bigscience/bloom-6b3")

model = FlaxBloomForCausalLM(config, _do_init=False, dtype=jnp.bfloat16, use_scan=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-6b3", use_fast=False)


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


def init_fn():
    input_shape = (1, 1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    return model.module.init(rng, input_ids, attention_mask, return_dict=False)


param_axes = jax.eval_shape(init_fn)["params_axes"]  # Axis names metadata

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

# shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)
# This will auto-magically run in mesh context
# params = shard_params(freeze(params))

init_params = partitioner.partition(init_fn, None, params_spec)

# This will auto-magically run in mesh context
params = init_params()

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


# print but only on the first node
def head_print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)


head_print("----------------------------")
start = time.time()
gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])
generated_text = tokenizer.batch_decode(np.asarray(gen_ids), skip_special_tokens=True)
head_print('Gen time:', time.time() - start)
head_print(generated_text)
head_print(len(generated_text))

head_print("----------------------------")
start = time.time()
gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])
generated_text = tokenizer.batch_decode(np.asarray(gen_ids), skip_special_tokens=True)
head_print('Gen time:', time.time() - start)
head_print(len(generated_text))

head_print("----------------------------")
start = time.time()
gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])
generated_text = tokenizer.batch_decode(np.asarray(gen_ids), skip_special_tokens=True)
head_print('Gen time:', time.time() - start)
head_print(len(generated_text))

head_print("----------------------------")
start = time.time()
gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])
generated_text = tokenizer.batch_decode(np.asarray(gen_ids), skip_special_tokens=True)
head_print('Gen time:', time.time() - start)
head_print(len(generated_text))
