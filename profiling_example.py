import jax
import jax.numpy as jnp

from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState
from t5x.checkpoints import Checkpointer

from transformers import AutoTokenizer

from bloom_inference.modeling_bloom import FlaxBloomForCausalLM, BloomConfig

#ckpt = "bigscience/bloom-350m"
#t5x_path = "gs://suraj-bloom-tpu-bucket/bloom-350m-scan-t5x/checkpoint_0"

"""ckpt = "bigscience/bloom-6b3"
t5x_path = "gs://suraj-tpu-bucket/bloom-6b3-scan-t5x-v3-8-pretrained/checkpoint_0"

max_input_length = 32
max_new_tokens = 32

model_parallel_submesh = (2, 2, 1, 2)"""

ckpt = "bigscience/bloom"
t5x_path = "gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x-final/checkpoint_0"
max_new_tokens = 32
max_input_length = 32
model_parallel_submesh = (1, 2, 4, 1)  # for v4-64


prompts = 8 * ['the cat sat on the mat']

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


config = BloomConfig.from_pretrained(ckpt)
model = FlaxBloomForCausalLM(config, _do_init=False, dtype=jnp.bfloat16, use_scan=True)


def init_state():
    input_shape = (1, 1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    attention_mask = jnp.ones_like(input_ids)
    rng = jax.random.PRNGKey(0)
    initial_vars = model.module.init(rng, input_ids, attention_mask, return_dict=False)
    return InferenceState.create(initial_vars)


state_shapes = jax.eval_shape(init_state)
partitioner = PjitPartitioner(
    model_parallel_submesh=model_parallel_submesh,
    logical_axis_rules=logical_axis_rules_full
)

params_spec = partitioner.get_mesh_axes(state_shapes).params
# Instantiate checkpointer
checkpointer = Checkpointer(
    state_shapes,
    partitioner,
    t5x_path,
    use_gda=True,
    restore_dtype=jnp.bfloat16,
    save_dtype=jnp.bfloat16
)

# load state
loaded_state = checkpointer.restore(path=t5x_path)
tokenizer = AutoTokenizer.from_pretrained(ckpt)
tokenizer.padding_side = "left"

max_length = max_input_length + max_new_tokens


def model_forward(params, input_ids, attention_mask):
    outputs = model(input_ids, attention_mask=attention_mask, params=params)
    return outputs


def sample_generate(params, input_ids, attention_mask):
    output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params, do_sample=True,
                                     num_beams=1, top_p=0.9, max_length=max_length).sequences
    return output_ids


p_forward = partitioner.partition(
            model_forward,
            in_axis_resources=(params_spec, None, None),
            out_axis_resources=None,
        )


p_sample_generate = partitioner.partition(
            sample_generate,
            in_axis_resources=(params_spec, None, None),
            out_axis_resources=None,
        )

inputs = tokenizer(prompts, return_tensors="jax", padding=True, truncation=True, max_length=max_input_length,
                        pad_to_multiple_of=max_input_length)

# This will auto-magically run in mesh context
forward_ids = p_forward(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])

# This will auto-magically run in mesh context
gen_ids = p_sample_generate(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])


with jax.profiler.trace(log_dir="/tmp/tensorboard/forward"):
    # generation
    forward_ids = p_forward(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])
    forward_ids.block_until_ready()

with jax.profiler.trace(log_dir="/tmp/tensorboard/generate"):
    # generation
    gen_ids = p_sample_generate(loaded_state.params, inputs["input_ids"], inputs["attention_mask"])
    gen_ids.block_until_ready()


generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

print(generated_text[0])
