import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState

from t5x import checkpoints
import flax
import chex

from bloom_inference.modeling_bloom import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer

ckpt = "sanchit-gandhi/bloom-350m-scan-t5x"

config = BloomConfig(n_layer=1)
model, params = FlaxBloomForCausalLM.from_pretrained(ckpt, _do_init=False, dtype=jnp.bfloat16, use_scan=True)
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

shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)

# This will auto-magically run in mesh context
params = shard_params(freeze(params))

# create frozen dict of model variables (params, params_axes), expected format of the .create method of InferenceState
model_variables = flax.core.freeze(***REMOVED***'params': params, 'params_axes': param_axes***REMOVED***)

# create InferenceState in .create method format (takes care of all attributes)
# TODO: flax_mutables & flax_mutables_axes required?
state = InferenceState.create(model_variables)

# Instantiate checkpointer
checkpointer = checkpoints.Checkpointer(state, partitioner, '/home/sanchitgandhi/dummy')

# save state -> save at step 0 will save to dir /checkpoint_0
checkpointer.save(state)

# load state
loaded_state = checkpointer.restore(path='/home/sanchitgandhi/dummy/checkpoint_0')

# Sanity checks
# 1. check params shapes equal
chex.assert_trees_all_equal_shapes(state.params, loaded_state.params), "params shapes not equal"
# 2. check params all equal
chex.assert_trees_all_equal(state.params, loaded_state.params), "params values not equal"
# 3. check params axes all equal
chex.assert_trees_all_equal(state.params_axes, loaded_state.params_axes), "params axes not equal"


