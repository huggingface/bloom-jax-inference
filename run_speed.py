import os
os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "34359738368"
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P

from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState
from t5x.checkpoints import Checkpointer

from bloom_inference.modeling_bloom import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer

# create a parser to get ckpt, path, max_len, input_len
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="bigscience/bloom")
parser.add_argument("--t5x_path", type=str, default="gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x/checkpoint_0")
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--input_len", type=int, default=64)
args = parser.parse_args()


ckpt = args.ckpt
path = args.t5x_path
max_len = args.max_len
input_len = args.input_len

config = BloomConfig.from_pretrained(ckpt)
model = FlaxBloomForCausalLM(config, _do_init=False, dtype=jnp.bfloat16, use_scan=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-350m", use_fast=False)

def head_print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)


# 2D parameter and activation partitioning
logical_axis_rules_full = [
    ('batch', None),
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

logical_axis_rules_palm = [
    ('batch', None),
    ('mlp', 'data'),
    ('heads', 'data'),
    ('vocab', 'data'),
    ('embed', 'model'),
    ('kv', None),
    ('length', None),
    ('layers', None),
    ('stack', None)
]


def init_state():
    input_shape = (1,1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    initial_vars = model.module.init(rng, input_ids, attention_mask, return_dict=False)
    return InferenceState.create(initial_vars)


state_shapes = jax.eval_shape(init_state)

# model_parallel_submesh = (2, 2, 2, 1), (2, 4, 1, 1), (2, 1, 4, 1) (1, 4, 2, 1) (1, 2, 4, 1)
model_parallel_submesh = (1, 2, 4, 1)
partitioner = PjitPartitioner(
    model_parallel_submesh=model_parallel_submesh,
    logical_axis_rules=logical_axis_rules_full
)
mesh_axes = partitioner.get_mesh_axes(state_shapes)
params_spec = mesh_axes.params

# Instantiate checkpointer
checkpointer = Checkpointer(state_shapes, partitioner, path, use_gda=True, restore_dtype=jnp.bfloat16, save_dtype=jnp.bfloat16)

# load state
head_print("Loading checkpoint")
loaded_state = checkpointer.restore(path=path)
head_print("Loading complete")

def generate(params, input_ids, attention_mask):
    output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
    return output_ids

p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, None, None),
    out_axis_resources=None
)

# setup for generation
tokenizer.padding_side = "left"
model.config.max_length = max_len
model.config.num_beams = 1
model.config.do_sample = True
model.config.top_p = 0.9


prompts = ["This is cool "] * 8
inputs = tokenizer(prompts, return_tensors="jax", padding="max_length", truncation=True, max_length=input_len)

# This will auto-magically run in mesh context
head_print("Compiling generate")
start = time.time()
gen_ids = p_generate(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])
generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
if jax.process_index() == 0:
    print("Compilation time:", time.time() - start)


start = time.time()
gen_ids = p_generate(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])
generated_text = tokenizer.batch_decode(gen_ids.local_shards[0].data, skip_special_tokens=False)
if jax.process_index() == 0:
    print("Generation time:", time.time() - start)

if jax.process_index() == 0:
    print(generated_text)