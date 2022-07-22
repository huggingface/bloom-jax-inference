import os
os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "44359738368"
import argparse
import time
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P

from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState
from t5x.checkpoints import Checkpointer

from bloom_inference.modeling_bloom import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer

t_start = time.time()

# create a parser to get ckpt, path, max_len, input_len
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="bigscience/bloom")
parser.add_argument("--t5x_path", type=str, default="gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x/checkpoint_0")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

num_tokens = 100

ckpt = args.ckpt
path = args.t5x_path
batch_size = args.batch_size


def head_print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)


config = BloomConfig.from_pretrained(ckpt)
model = FlaxBloomForCausalLM(config, _do_init=False, dtype=jnp.bfloat16, use_scan=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-350m", use_fast=False)


logical_axis_rules_palm = [
    ('batch', None),
    ('mlp', 'data'),
    ('heads', 'data'),
    ('vocab', 'data'),
    ('embed', 'model'),
    ('embed', 'model'),
    ('kv', None),
    ('length', None),
    ('layers', None),
    ('stack', None)
]


# get state shapes and initialize partitioner
def init_state():
    input_shape = (1,1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    initial_vars = model.module.init(rng, input_ids, attention_mask, return_dict=False)
    return InferenceState.create(initial_vars)

state_shapes = jax.eval_shape(init_state)

model_parallel_submesh = (1, 2, 4, 1)
partitioner = PjitPartitioner(
    model_parallel_submesh=model_parallel_submesh,
    logical_axis_rules=logical_axis_rules_palm
)
mesh_axes = partitioner.get_mesh_axes(state_shapes)
params_spec = mesh_axes.params


# Instantiate checkpointer
checkpointer = Checkpointer(state_shapes, partitioner, path, use_gda=True, restore_dtype=jnp.bfloat16, save_dtype=jnp.bfloat16)

# load state
head_print("Loading checkpoint")
loaded_state = checkpointer.restore(path=path)
head_print("Loading complete")

t_ready = time.time()

def generate(params, input_ids, attention_mask):
    output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
    return output_ids

# partition generate
p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, None, None),
    out_axis_resources=None
)


input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way"
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

# setup for generation
tokenizer.padding_side = "left"
model.config.max_length = num_tokens
model.config.min_length = num_tokens
model.config.do_sample = False

def run_generate():
    inputs = tokenizer(input_sentences, return_tensors="np", padding=True)
    gen_ids = p_generate(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])
    generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


# warm up
run_generate()

t_generate_start = time.time()
pairs = run_generate()
t_generate_span = time.time() - t_generate_start


# warm up
run_generate()

# benchmark
t0 = time.time()
cycles = 5
for i in range(cycles):
    _ = run_generate()

tokens_in_cycle = num_tokens * args.batch_size
througput = (time.time() - t0)/(cycles * tokens_in_cycle)

head_print(f"""
*** Performance stats:
Throughput per token including tokenize: {througput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {tokens_in_cycle} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
""")