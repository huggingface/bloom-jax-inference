import warnings

import jax
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


class Generator:
    def __init__(
            self,
            model_parallel_submesh=(1, 8, 1, 2),  # for v3-256
            ckpt="bigscience/bloom",
            t5x_path="gs://bloom-jax-us-central2-b/bloom-176B-scan-t5x-final/checkpoint_0",
            max_len=256,
            max_input_len=64,
    ):
        self.ckpt = ckpt
        self.path = t5x_path
        self.max_len = max_len
        self.max_input_len = max_input_len
        self.model_parallel_submesh = model_parallel_submesh

        config = BloomConfig.from_pretrained(self.ckpt)
        self.model = FlaxBloomForCausalLM(config, _do_init=False, dtype=jnp.bfloat16, use_scan=True)

        def init_state():
            input_shape = (1, 1)
            input_ids = jnp.zeros(input_shape, dtype="i4")
            attention_mask = jnp.ones_like(input_ids)
            rng = jax.random.PRNGKey(0)
            initial_vars = self.model.module.init(rng, input_ids, attention_mask, return_dict=False)
            return InferenceState.create(initial_vars)

        state_shapes = jax.eval_shape(init_state)
        self.partitioner = PjitPartitioner(
            model_parallel_submesh=self.model_parallel_submesh,
            logical_axis_rules=logical_axis_rules_full
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

    def load_model_and_params(self):
        # load state
        self.loaded_state = self.checkpointer.restore(path=self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.tokenizer.padding_side = "left"

        def greedy_generate(params, input_ids, attention_mask):
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, params=params, do_sample=False, num_beams=1).sequences
            return output_ids

        def sample_generate(params, input_ids, attention_mask):
            # TODO: top_k sampling, set to 0?
            output_ids = self.model.generate(input_ids, attention_mask=attention_mask, params=params, do_sample=True, num_beams=1, top_p=0.9).sequences
            return output_ids

        self.p_greedy_generate = self.partitioner.partition(
            greedy_generate,
            in_axis_resources=(self.params_spec, None, None),
            out_axis_resources=None,
        )

        self.p_sample_generate = self.partitioner.partition(
            sample_generate,
            in_axis_resources=(self.params_spec, None, None),
            out_axis_resources=None,
        )

    def generate(self, prompts, do_sample):
        if do_sample:
            return self.gen(prompts, self.p_sample_generate)
        else:
            return self.gen(prompts, self.p_greedy_generate)

    def gen(self, prompts, gen_fn):
        inputs = self.tokenizer(prompts, return_tensors="jax", padding=True, truncation=True, max_length=256, pad_to_multiple_of=128)
        max_length = inputs.input_ids.shape[-1] + 64
        self.model.config.max_length = max_length
        # This will auto-magically run in mesh context
        gen_ids = gen_fn(self.loaded_state.params, inputs["input_ids"], inputs["attention_mask"])

        generated_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return generated_text
