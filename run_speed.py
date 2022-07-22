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
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--input_len", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()


ckpt = args.ckpt
path = args.t5x_path
max_len = args.max_len
input_len = args.input_len
batch_size = args.batch_size

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
    # ('embed', 'data'),
    # ('joined_kv', 'data'),
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

def init_params():
    input_shape = (1,1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    return model.module.init(rng, input_ids, attention_mask, return_dict=False)["params"]


state_shapes = jax.eval_shape(init_state)

# model_parallel_submesh = (2, 2, 2, 1), (2, 4, 1, 1), (2, 1, 4, 1) (1, 4, 2, 1) (1, 2, 4, 1)
model_parallel_submesh = (1, 2, 4, 1)
partitioner = PjitPartitioner(
    model_parallel_submesh=model_parallel_submesh,
    logical_axis_rules=logical_axis_rules_palm
)
mesh_axes = partitioner.get_mesh_axes(state_shapes)
params_spec = mesh_axes.params


# p_init = partitioner.partition(
#     init_params,
#     in_axis_resources=None,
#     out_axis_resources=params_spec
# )

# head_print("init model")
# loaded_params = p_init()
# head_print("init complete")

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
model.config.min_length = max_len
model.config.num_beams = 1
model.config.do_sample = False
model.config.top_p = 0.9


prompts = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way"
]


def benchmark(bs, input_len, max_len):
    model.config.max_length = max_len
    model.config.min_length = max_len

    # prompts = ["Let's try a new text"] * bs
    prompts *= math.ceil(bs / len(prompts))
    inputs = tokenizer(prompts, return_tensors="jax", padding=True, truncation=True, max_length=input_len)

    # =====================================================================
    # This will auto-magically run in mesh context
    head_print("Compiling generate")
    start = time.time()
    gen_ids = p_generate(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])
    generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
    head_print(f"BS={bs} input_len={input_len} max_len={max_len} Compilation time:", time.time() - start)
    # =====================================================================


    # =====================================================================

    start = time.time()
    gen_ids = p_generate(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])
    generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
    head_print(f"BS={bs} input_len={input_len} max_len={max_len} Generation time:", time.time() - start)
    head_print("====================================================")
    # =====================================================================

# 64/64
# benchmark(bs=64, input_len=32, max_len=32+64)
# benchmark(bs=32, input_len=64, max_len=128)
# benchmark(bs=16, input_len=64, max_len=128)
# benchmark(bs=8, input_len=64, max_len=128)
# benchmark(bs=4, input_len=64, max_len=128)
# benchmark(bs=2, input_len=64, max_len=128)
# benchmark(bs=1, input_len=64, max_len=128)


# Compiling generate
# BS=32 input_len=64 max_len=128 Compilation time: 76.30435061454773
# BS=32 input_len=64 max_len=128 Generation time: 7.012312173843384
# ====================================================
# Compiling generate
# BS=16 input_len=64 max_len=128 Compilation time: 18.994248390197754
# BS=16 input_len=64 max_len=128 Generation time: 4.16681170463562
# ====================================================
# Compiling generate
# BS=8 input_len=64 max_len=128 Compilation time: 16.957380056381226
# BS=8 input_len=64 max_len=128 Generation time: 2.7345969676971436
# ====================================================
# Compiling generate
# BS=4 input_len=64 max_len=128 Compilation time: 70.61075520515442
# BS=4 input_len=64 max_len=128 Generation time: 2.2682857513427734
# ====================================================
# Compiling generate
# BS=2 input_len=64 max_len=128 Compilation time: 67.68023204803467
# BS=2 input_len=64 max_len=128 Generation time: 1.9246459007263184
# ====================================================
# Compiling generate
# BS=1 input_len=64 max_len=128 Compilation time: 66.19156551361084
# BS=1 input_len=64 max_len=128 Generation time: 1.7932021617889404
# ====================================================



## V3-256 (1, 8, 1, 2)
# Compiling generate
# BS=64 input_len=64 max_len=128 Compilation time: 141.89335203170776
# BS=64 input_len=64 max_len=128 Generation time: 20.445239782333374
# ====================================================
# Compiling generate
# BS=32 input_len=64 max_len=128 Compilation time: 21.499541997909546
# BS=32 input_len=64 max_len=128 Generation time: 11.290865182876587
# ====================================================
# Compiling generate
# BS=16 input_len=64 max_len=128 Compilation time: 16.689306020736694
# BS=16 input_len=64 max_len=128 Generation time: 6.655258655548096
# ====================================================
# Compiling generate
# BS=8 input_len=64 max_len=128 Compilation time: 14.247357606887817
# BS=8 input_len=64 max_len=128 Generation time: 4.338128089904785
# ====================================================
# Compiling generate
# BS=4 input_len=64 max_len=128 Compilation time: 45.78931999206543
# BS=4 input_len=64 max_len=128 Generation time: 3.388519525527954
# ====================================================
# Compiling generate
# BS=2 input_len=64 max_len=128 Compilation time: 42.81576061248779
# BS=2 input_len=64 max_len=128 Generation time: 3.039609909057617
# ====================================================
# Compiling generate
# BS=1 input_len=64 max_len=128 Compilation time: 42.88757038116455
# BS=1 input_len=64 max_len=128 Generation time: 2.68043