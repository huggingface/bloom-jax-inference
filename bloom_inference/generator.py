import warnings

import jax
jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P
from jax.experimental.compilation_cache import compilation_cache as cc

from t5x.partitioning import PjitPartitioner
from t5x.train_state import InferenceState
from t5x.checkpoints import Checkpointer

from transformers import AutoTokenizer

from bloom_inference.modeling_bloom import FlaxBloomForCausalLM, BloomConfig

cc.initialize_cache("~/jax_cache")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)

if jax.process_index() == 0:
    warnings.filterwarnings("default")


# print but only on the first node
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

# v3-32: standard rules suffer on v3-32, try PALMs
logical_axis_rules_palm = [
    ('batch', None),
    ('mlp', 'data'),
    ('heads', 'data'),
    ('vocab', 'data'),
    ('embed', 'model'),
    ('kv', None),
    # ('layer_norm_scale', 'model'),
    ('length', None),
    ('layers', None),
    ('stack', None)
]


class Generator:
    def __init__(
            self,
            model_parallel_submesh=(1, 8, 1, 2),  # for v3-256
            num_mp_partitions=None,
            ckpt="bigscience/bloom",
            t5x_path="gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x-final/checkpoint_0",
            max_len=256,
            max_input_len=64,
            unroll=1,
    ):
        self.ckpt = ckpt
        self.path = t5x_path
        self.max_len = max_len
        self.max_input_len = max_input_len
        self.model_parallel_submesh = model_parallel_submesh
        self.num_mp_partitions = num_mp_partitions
        self.unroll = unroll

        config = BloomConfig.from_pretrained(self.ckpt)
        self.model = FlaxBloomForCausalLM(config, _do_init=False, dtype=jnp.bfloat16, use_scan=True, unroll=self.unroll)

        def init_state():
            input_shape = (1, 1)
            input_ids = jnp.zeros(input_shape, dtype="i4")
            attention_mask = jnp.ones_like(input_ids)
            rng = jax.random.PRNGKey(0)
            initial_vars = self.model.module.init(rng, input_ids, attention_mask, return_dict=False)
            return InferenceState.create(initial_vars)

        state_shapes = jax.eval_shape(init_state)

        # v3-32: set num_partitions as opposed to submesh
        if self.num_mp_partitions:
            self.partitioner = PjitPartitioner(
                num_partitions=self.num_mp_partitions,
                logical_axis_rules=logical_axis_rules_palm,
            )
        else:
            self.partitioner = PjitPartitioner(
                model_parallel_submesh=self.model_parallel_submesh,
                logical_axis_rules=logical_axis_rules_palm,
            )
        self.params_spec = self.partitioner.get_mesh_axes(state_shapes).params

        # Instantiate checkpointer
        self.checkpointer = Checkpointer(
            state_shapes,
            self.partitioner,
            self.path,
            use_gda=True,
            restore_dtype=jnp.bfloat16,
            save_dtype=jnp.bfloat16
        )

        self.key = jax.random.PRNGKey(0)

    def load_model_and_params(self):
        # load state
        self.loaded_state = self.checkpointer.restore(path=self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.tokenizer.padding_side = "left"

        def greedy_generate(params, input_ids, attention_mask):
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, params=params, do_sample=False, num_beams=1).sequences
            return output_ids

        def sample_generate(params, input_ids, attention_mask, prng_key):
            # TODO: top_k sampling, set to 0?
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, params=params, do_sample=True, num_beams=1, top_p=0.9, prng_key=prng_key).sequences
            return output_ids

        # v3-32: Partition spec for DP
        self.p_greedy_generate = self.partitioner.partition(
            greedy_generate,
            in_axis_resources=(self.params_spec, P('data'), P('data')),
            out_axis_resources=P('data'),
        )

        # v3-32: Partition spec for DP
        self.p_sample_generate = self.partitioner.partition(
            sample_generate,
            in_axis_resources=(self.params_spec, P('data'), P('data'), None),
            out_axis_resources=P('data'),
        )

    def generate(self, prompts, do_sample):
        if do_sample:
            return self.gen(prompts, self.p_sample_generate)
        #else:
            #return self.gen(prompts, self.p_greedy_generate)

    def gen(self, prompts, gen_fn):
        try:
            inputs = self.tokenizer(prompts, return_tensors="jax", padding=True, truncation=True, max_length=self.max_input_len, pad_to_multiple_of=self.max_input_len)

            max_new_tokens = self.max_len - self.max_input_len
            max_length = inputs.input_ids.shape[-1] + int(max_new_tokens)
            self.model.config.max_length = max_length

            self.key, subkey = jax.random.split(self.key)
            # This will auto-magically run in mesh context
            gen_ids = gen_fn(self.loaded_state.params, inputs["input_ids"], inputs["attention_mask"], self.key)

            generated_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            return generated_text
        except Exception as e:
            head_print(e)
            return {"error": "something went wrong..."}
