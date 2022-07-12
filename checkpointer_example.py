import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P
from flax.core.frozen_dict import freeze

from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState
from t5x.checkpoints import Checkpointer

from bloom_inference.modeling_bloom import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer

jax.config.update('jax_parallel_functions_output_gda', True)

ckpt = "sanchit-gandhi/bloom-350m-scan-t5x"

config = BloomConfig(n_layer=1)
model, params = FlaxBloomForCausalLM.from_pretrained(ckpt, _do_init=False, dtype=jnp.bfloat16, use_scan=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-350m", use_fast=False)


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


def init_state():
    input_shape = (1,1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    initial_vars = model.module.init(rng, input_ids, attention_mask, return_dict=False)
    return InferenceState.create(initial_vars)


state_shapes = jax.eval_shape(init_state)

num_mp_partitions = 4
partitioner = PjitPartitioner(num_mp_partitions, logical_axis_rules=logical_axis_rules_full)
mesh_axes = partitioner.get_mesh_axes(state_shapes)
params_spec = mesh_axes.params

p_shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)

# This will auto-magically run in mesh context
params = p_shard_params(freeze(params))

# create frozen dict of model variables (params, params_axes), expected format of the .create method of InferenceState
model_variables = freeze(***REMOVED***'params': params, 'params_axes': state_shapes.params_axes***REMOVED***)

# create InferenceState in .create method format (takes care of all attributes)
# TODO: flax_mutables & flax_mutables_axes required?
state = InferenceState.create(model_variables)

# Instantiate checkpointer
path = "gs://suraj-tpu-bucket/bloom-6b3-scan-t5x-v3-8-pretrained"
checkpointer = Checkpointer(state_shapes, partitioner, path, use_gda=True, restore_dtype=jnp.bfloat16, save_dtype=jnp.bfloat16)

# save state -> save at step 0 will save to dir /checkpoint_0
checkpointer.save(state)

# load state
path = "gs://suraj-tpu-bucket/bloom-6b3-scan-t5x-v3-8-pretrained/checkpoint_0"
loaded_state = checkpointer.restore(path=path)

# Sanity checks
# 1. check params shapes equal
# chex.assert_trees_all_equal_shapes(state.params, loaded_state.params), "params shapes not equal"
# 2. check params all equal
# chex.assert_trees_all_equal(state.params, loaded_state.params), "params values not equal"
# 3. check params axes all equal
# chex.assert_trees_all_equal(state.params_axes, loaded_state.params_axes), "params axes not equal"

def generate(params, input_ids, attention_mask):
    output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
    return output_ids

p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, P("data"), P("data")),
    out_axis_resources=P("data")
)

# setup for generation
tokenizer.padding_side = "left"
model.config.max_length = 64
model.config.num_beams = 1
model.config.do_sample = False


prompts = ["This is cool "] * 4
inputs = tokenizer(prompts, return_tensors="jax", padding="max_length", truncation=True, max_length=16)
# This will auto-magically run in mesh context
gen_ids = p_generate(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])

generated_text = tokenizer.batch_decode(gen_ids.local_shards[0].data, skip_special_tokens=False)

if jax.process_index() == 0:
    print(generated_text)